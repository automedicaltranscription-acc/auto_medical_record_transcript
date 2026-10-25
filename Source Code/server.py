import json, os, io, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import bcrypt
import jwt
import pandas as pd
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Import the function from the STT pipeline.
# Wrapped in try/except because stt_pipeline.py also imports whisper & torch,
# which are heavy dependencies not installed on the lightweight Vercel deployment.
# The live demo only uses the Gemini-based SOAP extraction — Whisper is never
# called server-side (the browser handles transcription via the Web Speech API).
try:
    from stt_pipeline import diarize_and_extract_soap_from_text, _flatten
except ImportError as _e:
    print(f"[Config] stt_pipeline unavailable ({_e}) — running in Gemini-only mode.")
    # Provide stub implementations so the rest of the module loads cleanly.
    # diarize_and_extract_soap_from_text is called by /api/analyze, which still works
    # because that function only uses Gemini internally (not Whisper).
    import importlib, types as _types
    # Re-attempt importing just the functions we need from a stripped import path
    # by temporarily suppressing the whisper/torch requirement:
    import sys as _sys, os as _os
    _stt_mod = _types.ModuleType("stt_pipeline")
    # Minimal stubs — will be overwritten if partial import succeeds
    def _stub(*a, **kw): raise RuntimeError("stt_pipeline not available in this environment")
    _stt_mod.diarize_and_extract_soap_from_text = _stub
    _stt_mod._flatten = _stub
    _sys.modules.setdefault("stt_pipeline", _stt_mod)
    # Try a targeted import that skips the whisper/torch lines
    try:
        _stt_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "stt_pipeline.py")
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("stt_pipeline_gemini", _stt_path)
        # Can't easily skip the top-level imports — fall back to the Gemini-only approach
        # by importing google-genai directly in server.py for the analyze endpoint.
        from google import genai as _genai
        import json as _json

        def diarize_and_extract_soap_from_text(transcript_text: str):
            """Fallback: call Gemini directly for SOAP extraction (no Whisper)."""
            import os as _o
            client = _genai.Client(api_key=_o.environ.get("GEMINI_API_KEY"))
            prompt = (
                'You are a clinical documentation assistant. You receive a raw STT transcript '
                'from a doctor-patient consultation as a numbered list of timestamped segments. '
                'The conversation may be in English, Bahasa Indonesia, or a mix of both.\n\n'

                'Tasks:\n'
                '1. Identify Doctor vs Patient from conversational context, regardless of language.\n'
                '   In Indonesian: the doctor (dokter) asks clinical questions; the patient (pasien) describes symptoms.\n'
                '2. Label every segment as "Doctor" or "Patient".\n'
                '3. Extract a structured SOAP record using ONLY evidence that is explicitly present in the transcript.\n\n'

                '=== LANGUAGE RULE (STRICT) ===\n'
                'Detect the SINGLE dominant language of the dialogue FIRST, before writing anything.\n'
                '- If the dialogue is in English, every free-text field (summary, chief_complaint, symptoms, '
                'diagnosis, treatment_plan, etc.) MUST be written entirely in English. Do NOT translate any '
                'part of it into Bahasa Indonesia or any other language.\n'
                '- If the dialogue is in Bahasa Indonesia, every free-text field MUST be written entirely in '
                'Bahasa Indonesia. Do NOT leave any part of it in English.\n'
                '- Never mix languages within a single field or across fields.\n'
                '- For enum values, translate them directly (encounter_type: \'konsultasi_awal\', \'tindak_lanjut\', '
                '\'tinjauan_hasil\', \'lainnya\'; extraction_confidence: \'tinggi\', \'sedang\', \'rendah\') ONLY if the '
                'dialogue is in Indonesian. If English, use the English enum values shown in the schema.\n'
                '- Do NOT translate the JSON field names/keys -- those always stay in English exactly as shown.\n\n'

                '=== EVIDENCE RULE (STRICT -- READ CAREFULLY) ===\n'
                'This is the most important rule. Your job is transcription-to-structure, NOT diagnosis.\n'
                '- Only include a fact if it was explicitly said by the Doctor or Patient in the transcript.\n'
                '- Do NOT infer, guess, or "complete" the clinical picture using general medical knowledge.\n'
                '- If a field has no explicit supporting evidence, leave it as null (single-value fields) or '
                '[] (list fields). An empty field is the CORRECT output when the conversation does not cover '
                'that topic -- do not pad it to look complete.\n'
                '- Never write placeholder text like "not specified", "unknown", or "unclear" -- use null/[] instead.\n\n'
                '- assessment.diagnosis: Fill this ONLY if the Doctor explicitly states a diagnosis, working '
                'diagnosis, or clinical impression out loud (e.g. "this sounds like...", "I think you have...", '
                '"my diagnosis is..."). Pure history-taking conversations often end with NO diagnosis spoken -- '
                'that is normal and expected. In that case return diagnosis: []. Restating the chief complaint '
                '(e.g. "chest pain, unspecified etiology") is NOT a diagnosis -- do not invent one.\n'
                '- assessment.differential_diagnosis: Unlike diagnosis, this field DOES call for standard clinical '
                'reasoning, the way a real clinician writes it on a note. List plausible differentials a clinician '
                'would consider given the symptoms, history, and risk factors ACTUALLY STATED in the transcript '
                '(e.g. stated family history, symptom character, described risk factors). Ground every item in '
                'something specific that was said. If the Doctor explicitly rules something out or lists '
                'possibilities out loud, include those too.\n'
                '- symptoms[]: Include every symptom explicitly described by the patient (or doctor summarizing '
                'the patient) -- primary AND secondary. This is about not missing what WAS said, never about '
                'adding what was not.\n'
                '- additional_notes: Social/environmental factors explicitly mentioned (smoking, living situation, '
                'work/school exposures, travel). null if none discussed.\n'
                '- plan.follow_up: The Doctor\'s exact closing instructions, ONLY if actually given. null otherwise.\n'
                '- objective.physical_exam_findings[]: Observable signs described, and exam findings the doctor '
                'explicitly states. Do not infer findings that would be "expected" for a diagnosis but were never '
                'actually mentioned.\n\n'

                '=== DRUG NAME NORMALIZATION ===\n'
                'The transcript comes from speech-to-text and drug names are often phonetically garbled '
                '(e.g. "lacetipro" means Lisinopril, "metaforin" means Metformin). Normalize a garbled name to its '
                'correct spelling ONLY when a medication was clearly mentioned. If uncertain, write the corrected '
                'name followed by "[STT-corrected]". Never add a medication that was not mentioned at all.\n\n'

                '=== SELF-CHECK BEFORE RESPONDING ===\n'
                'Before outputting JSON, silently re-check each non-empty field against the transcript: "Can I '
                'point to the exact line that supports this?" If not, set it back to null/[]. Confirm every '
                'free-text field is in the single dominant language, and that diagnosis was not invented in a '
                'conversation that never stated one.\n\n'

                '=== OTHERS ===\n'
                "If a segment's text appears to end one speaker's turn and begin another's, split it into two diarized_segments entries and estimate a reasonable time split."
                'Return ONLY valid JSON (no markdown fences, no explanation). Schema:\n'
                '{\n'
                '  "diarized_segments": [{"id":0,"speaker":"Doctor","start":0.0,"end":5.0,"text":"..."}],\n'
                '  "summary": "2-3 sentence clinical summary using only facts stated in the transcript (in conversation language)",\n'
                '  "encounter_type": "initial_consultation|follow_up|results_review|other",\n'
                '  "additional_notes": "social/environmental context, or null if none",\n'
                '  "extraction_confidence": "high|medium|low",\n'
                '  "subjective": {"chief_complaint":"","symptoms":[],"symptom_onset":"","medical_history":[],"current_medications":[],"allergies":[]},\n'
                '  "objective": {"physical_exam_findings":[],"vital_signs":null},\n'
                '  "assessment": {"diagnosis":[],"differential_diagnosis":[]},\n'
                '  "plan": {"prescribed_medications":[],"treatment_plan":[],"investigations_ordered":[],"referrals":[],"follow_up":"closing instructions or null"}\n'
                '}\n'
                f"\nTranscript:\n{transcript_text}"
            )
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
                raw = raw.rsplit("```", 1)[0].strip()
            return _json.loads(raw)

        def _flatten(df):
            """Passthrough flatten — returns df unchanged in Gemini-only mode."""
            return df

    except Exception as _e2:
        print(f"[Config] Gemini-only fallback also failed: {_e2}")

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
load_dotenv()
JWT_SECRET   = os.environ.get("JWT_SECRET_KEY", "change-me-in-env")
JWT_ALGO     = "HS256"
JWT_EXPIRY   = timedelta(hours=12)       # re-login every 12 hours
USERS_FILE   = Path(__file__).parent / "users.json"
COOKIE_NAME  = "stt_session"

# ── Login toggle ──────────────────────────────────────────────────────────────
# Set ENABLE_LOGIN=false in .env to bypass the login page entirely.
# Set ENABLE_LOGIN=true  to enforce login with demo/premium accounts.
ENABLE_LOGIN = os.environ.get("ENABLE_LOGIN", "true").strip().lower() == "true"
print(f"[Config] Login required : {ENABLE_LOGIN}")

# Optional fixed expiry date for demo accounts (Vercel-compatible — no disk writes).
# Format: YYYY-MM-DD  e.g. "2026-08-21"
# Leave unset or empty for no expiry.
DEMO_EXPIRES_AT = os.environ.get("DEMO_EXPIRES_AT", "").strip()
if DEMO_EXPIRES_AT:
    print(f"[Config] Demo expires at : {DEMO_EXPIRES_AT}")
else:
    print("[Config] Demo expires at : (no expiry set)")

app = FastAPI(title="Medical STT Live API")

# Serve the static files (HTML, CSS, JS)
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ─────────────────────────────────────────────
#  User store helpers
# ─────────────────────────────────────────────

# Default users — pre-hashed with bcrypt.
# These are the credentials used if users.json does not exist yet
# (e.g. on a fresh Render deployment where the file isn't committed to git).
_DEFAULT_USERS = {
    "demo_user": {
        "password_hash": "$2b$12$vgZZk6.95gQ1EFre3e65b.7gr7I3ymTqNWoiSMh.tn3fwkVLfg0ki",
        "role": "demo",
        "first_login": None,
        "expires_days": 7,
    },
    "premium_user": {
        "password_hash": "$2b$12$JTgyms30l69cPe38qwT/XeehkOp9rzGg6BGeWx.I3Qm5k/ht2OlNu",
        "role": "premium",
        "first_login": None,
        "expires_days": None,
    },
}

# ── In-memory fallback for read-only filesystems (e.g. Vercel) ───────────────
# When users.json cannot be written to disk, we keep users in this dict.
# Login still works; state just doesn't persist across serverless cold starts.
_USERS_IN_MEMORY: bool = False
_users_cache: dict = {}

def _ensure_users_file() -> None:
    """Create users.json with default credentials if it doesn't exist.
    On read-only filesystems (Vercel), falls back to an in-memory store.
    """
    global _USERS_IN_MEMORY, _users_cache
    if not USERS_FILE.exists():
        print(f"[Auth] users.json not found — creating with default accounts at {USERS_FILE}")
        try:
            with open(USERS_FILE, "w", encoding="utf-8") as f:
                json.dump(_DEFAULT_USERS, f, indent=2)
            print("[Auth] users.json created [OK]")
        except OSError:
            # Read-only filesystem (Vercel) — use in-memory store instead
            print("[Auth] Read-only filesystem — using in-memory user store (Vercel mode).")
            _USERS_IN_MEMORY = True
            _users_cache = json.loads(json.dumps(_DEFAULT_USERS))  # deep copy
    else:
        print(f"[Auth] users.json found at {USERS_FILE}")

_ensure_users_file()  # Run at import time


def _load_users() -> dict:
    if _USERS_IN_MEMORY:
        return dict(_users_cache)  # return a copy
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: dict) -> None:
    global _users_cache
    if _USERS_IN_MEMORY:
        _users_cache = users  # update in-memory (won't persist across cold starts)
        return
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


# ─────────────────────────────────────────────
#  Auth helpers
# ─────────────────────────────────────────────
def _issue_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + JWT_EXPIRY,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def _verify_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.PyJWTError:
        return None


def get_current_user(request: Request) -> dict:
    """FastAPI dependency — returns the decoded JWT payload or raises 401.
    When ENABLE_LOGIN is False, always passes through as a guest user.
    """
    if not ENABLE_LOGIN:
        return {"sub": "guest", "role": "premium"}
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = _verify_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Session expired")
    return payload


# ─────────────────────────────────────────────
#  Pydantic models
# ─────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


class AnalyzeRequest(BaseModel):
    text: str


class CSVRequest(BaseModel):
    soap_data: dict


# ─────────────────────────────────────────────
#  Auth routes
# ─────────────────────────────────────────────
@app.get("/login")
async def login_page():
    with open(Path(__file__).parent / "static" / "login.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/login")
async def api_login(body: LoginRequest):
    users = _load_users()
    user_data = users.get(body.username)

    # --- Validate credentials ---
    if user_data is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    pw_bytes = body.password.encode("utf-8")
    stored_hash = user_data["password_hash"].encode("utf-8")
    if not bcrypt.checkpw(pw_bytes, stored_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    # --- Check demo account expiry ---
    # Expiry is controlled by the DEMO_EXPIRES_AT environment variable (YYYY-MM-DD).
    # This avoids writing to disk, which is not possible on Vercel's read-only filesystem.
    # Leave DEMO_EXPIRES_AT unset for no expiry.
    role = user_data["role"]
    if role == "demo" and DEMO_EXPIRES_AT:
        try:
            expiry_dt = datetime.fromisoformat(DEMO_EXPIRES_AT).replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) >= expiry_dt:
                # Return a generic 401 — indistinguishable from wrong credentials
                raise HTTPException(status_code=401, detail="Invalid username or password.")
        except ValueError:
            # Invalid date format in env var — ignore, no expiry enforced
            print(f"[Auth] WARNING: DEMO_EXPIRES_AT='{DEMO_EXPIRES_AT}' is not a valid date (expected YYYY-MM-DD). Expiry skipped.")

    # --- Issue token ---
    token = _issue_token(body.username, role)
    response = JSONResponse({"message": "Login successful", "role": role})
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=True,        # Vercel always serves over HTTPS
        samesite="lax",
        max_age=int(JWT_EXPIRY.total_seconds()),
    )
    return response


@app.post("/api/logout")
async def api_logout():
    response = JSONResponse({"message": "Logged out"})
    response.delete_cookie(COOKIE_NAME)
    return response


# ─────────────────────────────────────────────
#  Protected routes
# ─────────────────────────────────────────────
@app.get("/")
async def root(request: Request):
    """Redirect root → login if not authenticated, else → /app.
    When ENABLE_LOGIN is False, serves the app directly.
    """
    if not ENABLE_LOGIN:
        with open(Path(__file__).parent / "static" / "index.html", "r") as f:
            return HTMLResponse(content=f.read())
    token = request.cookies.get(COOKIE_NAME)
    if token and _verify_token(token):
        return RedirectResponse(url="/app")
    return RedirectResponse(url="/login")


@app.get("/app")
async def main_app(_user: dict = Depends(get_current_user)):
    with open(Path(__file__).parent / "static" / "index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/analyze")
async def analyze_text(request: AnalyzeRequest, _user: dict = Depends(get_current_user)):
    """
    Takes the raw transcript text from the frontend, sends it to Gemini for
    Diarization and SOAP extraction, and returns the structured JSON.
    """
    if not request.text.strip():
        return JSONResponse(status_code=400, content={"error": "Empty text provided"})

    result = diarize_and_extract_soap_from_text(request.text)

    if result is None:
        return JSONResponse(status_code=500, content={"error": "Failed to extract SOAP report."})

    return result


@app.post("/api/download_csv")
async def download_csv(request: CSVRequest, _user: dict = Depends(get_current_user)):
    """
    Takes the structured SOAP JSON, flattens it using our existing pipeline logic,
    and returns a raw CSV string for the frontend to download.
    """
    soap_dict = request.soap_data
    # Remove diarized_segments for the flat CSV report
    soap_clean = {k: v for k, v in soap_dict.items() if k != "diarized_segments"}

    # Flatten using pandas (same logic as in save_results)
    flat = pd.json_normalize(soap_clean, sep=".")
    flat = _flatten(flat)

    # We don't have an audio file name for live STT, so use 'live_recording'
    flat.insert(0, "audio_file", "live_recording")

    # Convert to CSV string
    output = io.StringIO()
    flat.to_csv(output, index=False)

    return PlainTextResponse(content=output.getvalue(), media_type="text/csv")


# ─────────────────────────────────────────────
#  ICD-10 Mapping (Jev System One)
# ─────────────────────────────────────────────
try:
    from jev_icd10_mapper import map_soap_to_icd10
except ImportError as _je:
    print(f"[Config] jev_icd10_mapper unavailable ({_je}) — using fallback stub.")
    def map_soap_to_icd10(soap_data, force_mock=False):
        return {
            "code": "R07",
            "description": "Precordial / chest pain unspecified",
            "confidence": 0.50,
            "status": "FLAG_FOR_REVIEW",
            "mode": "FALLBACK",
            "top_probabilities": {"R07": 0.50}
        }


class ICD10Request(BaseModel):
    soap_data: dict


@app.post("/api/map-icd10")
async def api_map_icd10(request: ICD10Request, _user: dict = Depends(get_current_user)):
    """
    Takes the structured SOAP JSON, evaluates diagnostic criteria against
    the PERKI ICD-10 cardiovascular table using Jev System One model,
    and returns the predicted code, confidence, and review status.
    """
    if not request.soap_data:
        return JSONResponse(status_code=400, content={"error": "Empty SOAP data provided."})

    try:
        result = map_soap_to_icd10(request.soap_data)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"ICD-10 mapping failed: {str(e)}"})


# ─────────────────────────────────────────────
#  API: return current user info (for the frontend)
# ─────────────────────────────────────────────
@app.get("/api/me")
async def me(user: dict = Depends(get_current_user)):
    return {"username": user["sub"], "role": user["role"]}


if __name__ == "__main__":
    import uvicorn
    print("Starting Medical STT Live Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
