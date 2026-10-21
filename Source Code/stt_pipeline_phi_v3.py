"""
stt_pipeline_phi_v3.py — Phi-4-mini SOAP pipeline with Pydantic schema enforcement
====================================================================================

v3   — Schema-constrained decoding, short prompt, one-shot example, single-pass diarization.
v3.1 — Fixed three extraction gaps found on CAR0001:
         • Mixed-segment handling: Whisper often bundles doctor+patient speech into one
           segment. v3 was collapsing patient answers into fabricated "No."  Now the
           prompt explicitly tells the model to use verbatim text and split carefully.
         • symptoms[] now includes the chief complaint as the first entry.
         • medical_history[] now explicitly covers family history.
         • additional_notes now explicitly captures ALL social history.
         • Few-shot example updated to demonstrate a mixed-segment split and social history.

Usage:
    python stt_pipeline_phi_v3.py [segments_json] [--output output.json]

Requirements:
    - Ollama running locally (ollama serve)
    - phi4-mini pulled (ollama pull phi4-mini)
    - requests, pydantic (both already in requirements.txt)
"""

import json
import os
import sys
import time
from typing import Optional

import requests
from pydantic import BaseModel, Field


# ============================================================================
# Configuration
# ============================================================================

OLLAMA_URL     = os.getenv("OLLAMA_URL",     "http://127.0.0.1:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",   "phi4-mini")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))


# ============================================================================
# Pydantic schema
# Every field has an explicit type. Ollama uses this for grammar sampling.
# ============================================================================

class DiarizedSegment(BaseModel):
    id: int
    speaker: str = Field(description="Doctor or Patient")
    start: float
    end: float
    text: str


class Subjective(BaseModel):
    chief_complaint: Optional[str] = None
    symptoms: list[str] = Field(default_factory=list)
    symptom_onset: Optional[str] = None
    medical_history: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)


class Objective(BaseModel):
    physical_exam_findings: list[str] = Field(default_factory=list)
    vital_signs: Optional[str] = None


class Assessment(BaseModel):
    diagnosis: list[str] = Field(default_factory=list)
    differential_diagnosis: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    prescribed_medications: list[str] = Field(default_factory=list)
    treatment_plan: list[str] = Field(default_factory=list)
    investigations_ordered: list[str] = Field(default_factory=list)
    referrals: list[str] = Field(default_factory=list)
    follow_up: Optional[str] = None


class SOAPReport(BaseModel):
    diarized_segments: list[DiarizedSegment] = Field(default_factory=list)
    summary: Optional[str] = None
    encounter_type: str = "initial_consultation"
    additional_notes: Optional[str] = None
    extraction_confidence: str = Field(default="medium", description="high, medium, or low")
    subjective: Subjective = Field(default_factory=Subjective)
    objective: Objective = Field(default_factory=Objective)
    assessment: Assessment = Field(default_factory=Assessment)
    plan: Plan = Field(default_factory=Plan)


# ============================================================================
# System prompt — kept short and direct for a 3.8B model
# ============================================================================

SYSTEM_PROMPT = """You are a medical scribe. From the doctor-patient transcript, extract ONLY facts explicitly stated.

Rules:
1. Include ONLY what was explicitly said. Do NOT infer, guess, or use medical knowledge.
2. Doctor questions alone are NOT patient facts. Only patient answers/statements count.
3. CRITICAL — Mixed segments: Each Whisper segment may contain BOTH doctor and patient speech
   concatenated together. Split them carefully into separate diarized_segments entries.
   Copy the actual spoken text verbatim — do NOT fabricate short answers like "No." or "Yes."
   if the real answer was longer. Every patient statement must be preserved.
4. symptoms[]: the chief complaint is ALWAYS the first entry, followed by all associated symptoms.
5. medical_history[]: include BOTH personal past conditions AND family history
   (e.g. "Father had heart attack at age 45", "Father may have cholesterol problems").
6. additional_notes: capture ALL social history in one string — smoking, alcohol, drug use,
   diet, exercise, living situation. Use null ONLY if none of these were discussed.
7. If information is absent, use null or [].
8. diagnosis[]: ONLY if the doctor explicitly states a diagnosis out loud. Otherwise [].
9. differential_diagnosis[]: list clinically plausible differentials grounded in stated symptoms/history.
10. Preserve the conversation language (English transcript -> English output, Indonesian -> Indonesian).
11. For STT-garbled drug names, normalize to correct spelling and append [STT-corrected].
12. Label each segment "Doctor" or "Patient" based on conversational role.
13. Return raw JSON only — no explanation, no markdown."""


# ============================================================================
# One-shot example injected into the user message
# ============================================================================

FEW_SHOT_EXAMPLE = """EXAMPLE INPUT (note segment 2 contains BOTH doctor and patient speech mixed):
Segment 0 [0.0s->4.0s]: What brings you in today?
Segment 1 [4.0s->9.0s]: I have a bad headache since this morning, throbbing on the right side.
Segment 2 [9.0s->14.0s]: Do you have nausea? Yes, and I vomited once. Any fever? No fever.
Segment 3 [14.0s->22.0s]: Do you smoke? Yes I smoke about a pack a day for the past 10 years. Okay do you drink? I have about two drinks a day.

EXAMPLE OUTPUT — showing how to split mixed segments and capture social history:
{
  "diarized_segments": [
    {"id":0,"speaker":"Doctor","start":0.0,"end":4.0,"text":"What brings you in today?"},
    {"id":1,"speaker":"Patient","start":4.0,"end":9.0,"text":"I have a bad headache since this morning, throbbing on the right side."},
    {"id":2,"speaker":"Doctor","start":9.0,"end":10.5,"text":"Do you have nausea?"},
    {"id":2,"speaker":"Patient","start":10.5,"end":12.0,"text":"Yes, and I vomited once."},
    {"id":2,"speaker":"Doctor","start":12.0,"end":13.0,"text":"Any fever?"},
    {"id":2,"speaker":"Patient","start":13.0,"end":14.0,"text":"No fever."},
    {"id":3,"speaker":"Doctor","start":14.0,"end":15.5,"text":"Do you smoke?"},
    {"id":3,"speaker":"Patient","start":15.5,"end":17.5,"text":"Yes I smoke about a pack a day for the past 10 years."},
    {"id":3,"speaker":"Doctor","start":17.5,"end":19.0,"text":"Okay do you drink?"},
    {"id":3,"speaker":"Patient","start":19.0,"end":22.0,"text":"I have about two drinks a day."}
  ],
  "summary": "Patient presents with right-sided throbbing headache since this morning with nausea and one episode of vomiting. No fever reported. Smoker (1 pack/day, 10 years) and drinks approximately 2 drinks/day.",
  "encounter_type": "initial_consultation",
  "additional_notes": "Smoker: 1 pack/day for 10 years. Alcohol: approximately 2 drinks/day.",
  "extraction_confidence": "high",
  "subjective": {
    "chief_complaint": "headache",
    "symptoms": ["throbbing right-sided headache since this morning", "nausea", "vomited once"],
    "symptom_onset": "this morning",
    "medical_history": [],
    "current_medications": [],
    "allergies": []
  },
  "objective": {"physical_exam_findings": [], "vital_signs": null},
  "assessment": {
    "diagnosis": [],
    "differential_diagnosis": ["migraine", "tension headache"]
  },
  "plan": {
    "prescribed_medications": [], "treatment_plan": [],
    "investigations_ordered": [], "referrals": [], "follow_up": null
  }
}

NOW process the real transcript below:
"""


# ============================================================================
# Helpers
# ============================================================================

def load_segments(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Segments file must contain a non-empty JSON list.")
    return data


def format_transcript(segments: list[dict]) -> str:
    return "\n".join(
        f'Segment {s.get("id", i)} '
        f'[{float(s.get("start", 0)):.1f}s->{float(s.get("end", 0)):.1f}s]: '
        f'{str(s.get("text", "")).strip()}'
        for i, s in enumerate(segments)
    )


def ollama_chat_with_schema(system: str, user: str, schema: dict) -> str:
    """
    Call Ollama /api/chat with a full JSON Schema in the `format` field.
    Grammar-sampled decoding: the model cannot output tokens violating the schema.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "format": schema,   # full JSON Schema, not just "json"
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }

    url = OLLAMA_URL.rstrip("/") + "/api/chat"
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()

    content = r.json().get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError("Ollama returned an empty response.")
    return content


# ============================================================================
# Main extraction function
# ============================================================================

def extract_soap(segments: list[dict]) -> SOAPReport:
    transcript = format_transcript(segments)

    user_msg = FEW_SHOT_EXAMPLE + "TRANSCRIPT:\n" + transcript

    print(f"[Phi-v3] Sending {len(segments)} segments to {OLLAMA_MODEL} "
          f"(schema-constrained, temperature=0, ctx={OLLAMA_NUM_CTX})...")

    t0 = time.time()
    raw = ollama_chat_with_schema(
        SYSTEM_PROMPT,
        user_msg,
        SOAPReport.model_json_schema(),
    )
    elapsed = time.time() - t0

    print(f"[Phi-v3] Response in {elapsed:.1f}s")
    print(f"[Phi-v3] Raw output (first 300 chars): {raw[:300]}")

    # Pydantic validates and coerces. With schema enforcement this rarely fails.
    report = SOAPReport.model_validate_json(raw)
    return report


# ============================================================================
# Public API — same function names as other pipelines for easy swapping
# ============================================================================

def diarize_and_extract_soap(segments: list[dict]) -> dict:
    """For segment-based pipelines (e.g. batch_soap_analysis.py)."""
    return extract_soap(segments).model_dump()


def diarize_and_extract_soap_from_text(raw_text: str) -> dict:
    """
    For live-STT mode (server.py /api/analyze).
    Wraps raw transcript text into a single segment and runs extraction.
    """
    segments = [{"id": 0, "start": 0.0, "end": 0.0, "text": raw_text}]
    return extract_soap(segments).model_dump()


# ============================================================================
# CLI
# ============================================================================

def run(segments_file: str, output_file: str) -> None:
    start = time.time()

    print("=" * 60)
    print("Phi-4-mini v3: schema-constrained SOAP extraction")
    print("=" * 60)

    segments = load_segments(segments_file)
    print(f"[Input] Loaded {len(segments)} segments from {segments_file}")

    report = extract_soap(segments)

    output = {
        "model": OLLAMA_MODEL,
        "pipeline": "phi_v3_schema_constrained",
        "soap": report.model_dump(),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[Output] Written to: {output_file}")
    print(f"[Total]  {time.time() - start:.1f}s")

    s = report.subjective
    a = report.assessment
    print("\n--- Quick summary ---")
    print(f"  Chief complaint : {s.chief_complaint}")
    print(f"  Symptoms        : {s.symptoms}")
    print(f"  Onset           : {s.symptom_onset}")
    print(f"  Diagnosis       : {a.diagnosis}")
    print(f"  Differentials   : {a.differential_diagnosis}")
    print(f"  Confidence      : {report.extraction_confidence}")
    print(f"  Summary         : {report.summary}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phi-4-mini v3: schema-constrained SOAP extraction"
    )
    parser.add_argument(
        "segments_file",
        nargs="?",
        default="audio_sample/audio_recordings/Audio_Recordings/CAR0001_segments.json",
    )
    parser.add_argument("--output", default="car0001_phi_v3_soap.json")
    args = parser.parse_args()

    try:
        run(args.segments_file, args.output)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
