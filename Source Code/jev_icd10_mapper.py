"""
jev_icd10_mapper.py
===================
Maps clinical SOAP reports to ICD-10 cardiovascular codes using TypeSafe AI's
Jev (System One model) with calibrated confidence scoring and confidence-gating.

Features:
- Dual Mode:
    1. Live API Mode: Calls https://api.typesafe.ai/v1/systemone when TYPESAFE_API_KEY is set.
    2. Mock / Dry-Run Mode: Simulates Jev's Choice primitive (probabilities + confidence)
       locally without needing an API key or credit balance.
- Hierarchical / Constrained Choice:
    Uses the official PERKI cardiology table (icd10_cardiovascular.csv, 367 entries).
- Confidence-Gating (Selective Prediction from Paper 1):
    Scores >= threshold (default 0.60) -> AUTO_ACCEPT
    Scores < threshold               -> FLAG_FOR_REVIEW (indicates incomplete notes/unconfirmed diagnosis)

Usage:
    # Test on the 5 cardiology cases (CAR0001 - CAR0005)
    python jev_icd10_mapper.py --test-car-cases

    # Test single case with mock mode forced
    python jev_icd10_mapper.py --mock --case CAR0003

    # Run on custom SOAP text or JSON file
    python jev_icd10_mapper.py --soap-file path/to/soap.json
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Paths
SOURCE_DIR = Path(__file__).parent
ENV_FILE = SOURCE_DIR / ".env"
ICD10_CSV = SOURCE_DIR / "icd10_cardiovascular.csv"
CLEAN_TRANS_DIR = SOURCE_DIR.parent / "audio_sample" / "audio_recordings" / "Clean_Transcripts"
OUTPUT_DIR = SOURCE_DIR / "jev_output"

# API Endpoint & Model
JEV_API_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-1.13.0"
CONFIDENCE_THRESHOLD = 0.60


def load_api_key():
    """Load TYPESAFE_API_KEY from environment or .env file."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if key and key.strip() and not key.startswith("jv_mock"):
        return key.strip()
    if ENV_FILE.exists():
        with open(ENV_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("TYPESAFE_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val and not val.startswith("jv_mock"):
                        return val
    return None


def load_cardiovascular_table():
    """Load categories and codes from icd10_cardiovascular.csv."""
    if not ICD10_CSV.exists():
        raise FileNotFoundError(f"Missing knowledge base: {ICD10_CSV}")
    
    categories = {}
    subcodes = {}
    with open(ICD10_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            code = (r.get("code") or "").strip()
            desc = (r.get("description") or "").strip()
            level = (r.get("level") or "").strip()
            if not code or not desc:
                continue
            if level == "category":
                categories[code] = desc
            elif level == "code":
                subcodes[code] = desc

    return categories, subcodes


# ===========================================================================
# Bilingual Clinical Concept Normalization & Expansion (Indonesian -> English)
# ===========================================================================

# Curated ontology of Indonesian cardiology symptoms, findings, and diagnoses
# mapped to canonical English ICD-10 medical terminology.
# Ordered by descending specificity/length so multi-word phrases match before substrings.
INDONESIAN_TO_ENGLISH_CARDIOLOGY = [
    # 1. Gagal Jantung / Heart Failure (I50)
    ("gagal jantung kongestif kronik", "chronic congestive heart failure"),
    ("gagal jantung kongestif akut", "acute congestive heart failure"),
    ("gagal jantung kongestif", "congestive heart failure"),
    ("gagal jantung kiri", "left ventricular failure left heart failure"),
    ("gagal jantung kanan", "right ventricular failure right heart failure"),
    ("gagal jantung akut", "acute heart failure"),
    ("gagal jantung kronik", "chronic heart failure"),
    ("gagal jantung", "heart failure congestive heart failure"),
    ("edema paru akut", "acute pulmonary edema"),
    ("edema paru", "pulmonary edema pulmonary congestion"),
    ("syok kardiogenik", "cardiogenic shock"),

    # 2. Sindrom Koroner Akut & Infark Miokard (I20 - I25)
    ("infark miokard akut transmural", "acute transmural myocardial infarction STEMI"),
    ("infark miokard akut subendokardial", "acute subendocardial myocardial infarction NSTEMI"),
    ("infark miokard akut", "acute myocardial infarction STEMI NSTEMI"),
    ("infark miokard", "myocardial infarction"),
    ("serangan jantung", "acute myocardial infarction heart attack"),
    ("sindrom koroner akut", "acute coronary syndrome"),
    ("stemi", "st-elevation myocardial infarction transmural infarction"),
    ("nstemi", "non-st-elevation myocardial infarction subendocardial infarction"),
    ("angina pektoris tidak stabil", "unstable angina pectoris"),
    ("angina pektoris stabil", "stable angina pectoris"),
    ("angina pektoris", "angina pectoris ischemic heart disease"),
    ("angina", "angina pectoris ischemic heart disease"),
    ("penyakit jantung koroner", "coronary artery disease ischemic heart disease"),
    ("penyakit jantung iskemik", "ischemic heart disease"),
    ("aterosklerosis koroner", "coronary atherosclerosis artery disease"),

    # 3. Hipertensi (I10 - I15)
    ("krisis hipertensi", "hypertensive crisis emergency"),
    ("hipertensi emergensi", "hypertensive emergency"),
    ("hipertensi urgensi", "hypertensive urgency"),
    ("penyakit jantung hipertensi", "hypertensive heart disease"),
    ("hipertensi esensial", "essential primary hypertension"),
    ("hipertensi primer", "essential primary hypertension"),
    ("tekanan darah tinggi", "essential primary hypertension high blood pressure"),
    ("darah tinggi", "essential primary hypertension high blood pressure"),
    ("hipertensi", "essential primary hypertension"),
    ("hipotensi", "hypotension low blood pressure"),

    # 4. Aritmia & Gangguan Konduksi (I44 - I49)
    ("fibrilasi atrium", "atrial fibrillation flutter"),
    ("flutter atrium", "atrial flutter"),
    ("fibrilasi ventrikel", "ventricular fibrillation flutter"),
    ("takikardia ventrikel", "ventricular tachycardia"),
    ("takikardia supraventrikular", "supraventricular tachycardia"),
    ("aritmia jantung", "cardiac arrhythmia dysrhythmia"),
    ("aritmia", "cardiac arrhythmia dysrhythmia"),
    ("disritmia", "cardiac arrhythmia dysrhythmia"),
    ("blok av total", "complete atrioventricular heart block third degree"),
    ("blok av derajat dua", "second degree atrioventricular heart block"),
    ("blok av derajat satu", "first degree atrioventricular heart block"),
    ("blok av", "atrioventricular heart block conduction disorder"),
    ("bradikardia", "bradycardia slow heart rate conduction disorder"),
    ("takikardia", "tachycardia fast heart rate"),
    ("ekstrasistol ventrikel", "premature ventricular contractions extrasystole"),
    ("ekstrasistol atrium", "premature atrial contractions extrasystole"),

    # 5. Penyakit Katup, Miokard, Perikard & Lainnya (I05 - I09, I30 - I42)
    ("perikarditis akut", "acute pericarditis pericardium"),
    ("perikarditis", "acute pericarditis pericardium"),
    ("miokarditis akut", "acute myocarditis myocardium"),
    ("miokarditis", "acute myocarditis myocardium"),
    ("endokarditis infektif", "infective endocarditis"),
    ("endokarditis", "infective endocarditis"),
    ("kardiomiopati dilatasi", "dilated cardiomyopathy"),
    ("kardiomiopati hipertrofik", "hypertrophic cardiomyopathy"),
    ("kardiomiopati", "cardiomyopathy"),
    ("penyakit katup jantung", "valvular heart disease"),
    ("penyakit jantung rematik", "rheumatic heart disease"),
    ("demam rematik", "rheumatic fever"),
    ("stenosis mitral", "mitral valve stenosis"),
    ("regurgitasi mitral", "mitral valve regurgitation"),
    ("stenosis aorta", "aortic valve stenosis"),
    ("regurgitasi aorta", "aortic valve regurgitation"),
    ("kardiomegali", "cardiomegaly cardiac enlargement"),
    ("diseksi aorta", "aortic dissection"),
    ("aneurisma aorta", "aortic aneurysm"),

    # 6. Gejala & Keluhan Pasien (Subjective / Symptoms)
    ("nyeri dada sebelah kiri", "left-sided chest pain precordial pain"),
    ("nyeri dada kiri", "left-sided chest pain precordial pain"),
    ("nyeri dada prekordial", "precordial chest pain"),
    ("nyeri dada seperti tertekan", "crushing chest pain pressure sensation angina"),
    ("nyeri dada seperti tertindih", "crushing chest pain pressure sensation angina"),
    ("nyeri dada menjalar ke lengan kiri", "chest pain radiating to left arm"),
    ("nyeri dada menjalar", "radiating chest pain"),
    ("nyeri dada", "chest pain precordial pain throat and chest pain"),
    ("sakit dada", "chest pain precordial pain"),
    ("dada terasa tertekan", "chest pressure crushing sensation heaviness"),
    ("dada tertekan", "chest pressure crushing sensation heaviness"),
    ("dada terasa berat", "heavy chest pressure sensation"),
    ("dada tertindih beban berat", "crushing chest pain pressure"),
    ("dada tertindih", "crushing chest pain pressure"),
    ("sesak napas saat berbaring", "orthopnea dyspnea when lying flat"),
    ("sesak nafas saat berbaring", "orthopnea dyspnea when lying flat"),
    ("sesak saat tidur", "orthopnea dyspnea lying flat paroxysmal nocturnal dyspnea"),
    ("sesak napas malam hari", "paroxysmal nocturnal dyspnea waking up breathless"),
    ("sesak napas saat aktivitas", "dyspnea on exertion shortness of breath"),
    ("sesak napas", "dyspnea shortness of breath difficulty breathing"),
    ("sesak nafas", "dyspnea shortness of breath difficulty breathing"),
    ("sulit bernapas", "dyspnea shortness of breath difficulty breathing"),
    ("susah bernapas", "dyspnea shortness of breath difficulty breathing"),
    ("ortopnea", "orthopnea dyspnea lying flat"),
    ("kaki bengkak", "edema leg swelling peripheral edema ankles"),
    ("bengkak pada kaki", "edema leg swelling peripheral edema ankles"),
    ("bengkak di pergelangan kaki", "edema leg swelling ankle swelling"),
    ("bengkak kedua kaki", "bilateral leg edema peripheral edema"),
    ("bengkak tungkai", "peripheral edema leg swelling"),
    ("jantung berdebar-debar", "palpitations racing heart tachycardia"),
    ("jantung berdebar", "palpitations racing heart tachycardia"),
    ("berdebar-debar", "palpitations racing heart tachycardia"),
    ("berdebar", "palpitations racing heart tachycardia"),
    ("deg-degan", "palpitations racing heart tachycardia"),
    ("mudah lelah", "fatigue exercise intolerance weakness"),
    ("cepat lelah", "fatigue exercise intolerance weakness"),
    ("kelelahan", "fatigue exercise intolerance"),
    ("keringat dingin", "diaphoresis cold sweat"),
    ("pingsan", "syncope loss of consciousness fainting"),
    ("sinkop", "syncope loss of consciousness fainting"),
    ("pusing melayang", "presyncope lightheadedness dizziness"),
    ("pusing kleyengan", "presyncope lightheadedness dizziness"),
    ("pusing", "dizziness lightheadedness"),
    ("mual muntah", "nausea vomiting"),
    ("mual", "nausea"),
    ("batuk darah", "hemoptysis coughing blood"),

    # 7. Temuan Pemeriksaan Fisik (Objective Findings)
    ("ronki basah halus basal", "fine wet crackles rales pulmonary congestion basal"),
    ("ronki basah halus", "fine wet crackles rales pulmonary congestion"),
    ("ronki basah", "crackles rales pulmonary congestion"),
    ("ronki", "crackles rales pulmonary congestion"),
    ("ronkhi", "crackles rales pulmonary congestion"),
    ("suara napas basah", "pulmonary rales crackles"),
    ("edema pretibial bilateral", "bilateral pretibial pitting edema peripheral edema"),
    ("edema pretibial", "pretibial pitting edema peripheral edema"),
    ("pitting edema", "pitting edema peripheral edema"),
    ("peningkatan jvp", "elevated jugular venous pressure jvp"),
    ("jvp meningkat", "elevated jugular venous pressure jvp"),
    ("tekanan vena jugularis meningkat", "elevated jugular venous pressure jvp"),
    ("bising jantung sistolik", "systolic cardiac murmur heart murmur"),
    ("bising jantung diastolik", "diastolic cardiac murmur heart murmur"),
    ("bising jantung", "cardiac murmur heart murmur"),
    ("murmur", "cardiac murmur heart murmur"),
    ("gallop s3", "s3 gallop third heart sound ventricular gallop"),
    ("gallop s4", "s4 gallop fourth heart sound atrial gallop"),
    ("irama gallop", "gallop rhythm third heart sound"),
    ("gallop", "gallop rhythm third heart sound"),
    ("iktus kordis bergeser", "displaced cardiac apex PMI cardiomegaly"),
    ("batas jantung melebar", "cardiomegaly enlarged cardiac silhouette")
]


def flatten_soap_field(val):
    """Recursively flatten strings, lists, or dicts into a clean space-separated string."""
    if not val:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        items = [flatten_soap_field(x) for x in val if x]
        return " ".join(items).strip()
    if isinstance(val, dict):
        items = [flatten_soap_field(v) for v in val.values() if v]
        return " ".join(items).strip()
    return str(val).strip()


def normalize_and_expand_bilingual_soap(soap_data):
    """
    Normalizes SOAP text that may be in Indonesian or English.
    Translates & augments clinical terms into canonical English medical concepts
    so Jev System One and ICD-10 candidate criteria match with high calibration.
    """
    if not isinstance(soap_data, dict):
        raw = str(soap_data or "")
        return {
            "assessment": raw,
            "subjective": "",
            "objective": "",
            "plan": "",
            "raw_assessment": raw,
            "raw_subjective": "",
            "raw_objective": "",
            "clinical_concepts_en": [],
            "detected_language": "en"
        }

    raw_assessment = flatten_soap_field(soap_data.get("assessment"))
    raw_subjective = flatten_soap_field(soap_data.get("subjective"))
    raw_objective = flatten_soap_field(soap_data.get("objective"))
    raw_plan = flatten_soap_field(soap_data.get("plan"))

    combined_raw = f"{raw_assessment} {raw_subjective} {raw_objective}".lower()

    # Match Indonesian cardiology ontology
    matched_concepts = []
    matched_id_terms = []
    for id_term, en_term in INDONESIAN_TO_ENGLISH_CARDIOLOGY:
        if id_term in combined_raw:
            matched_concepts.append(en_term)
            matched_id_terms.append(id_term)

    # Detect language: check for Indonesian medical terms or common stopwords
    id_stopwords = {"dan", "yang", "pada", "saat", "dengan", "pasien", "sejak", "tidak", "mengeluh", "keluhan", "ada"}
    words = set(re.findall(r"\w+", combined_raw))
    is_indonesian = bool(matched_id_terms or (words & id_stopwords))

    def expand_field(field_text):
        if not field_text:
            return ""
        text_lower = field_text.lower()
        field_matches = []
        for id_term, en_term in INDONESIAN_TO_ENGLISH_CARDIOLOGY:
            if id_term in text_lower:
                field_matches.append(en_term)
        if field_matches:
            unique_en = list(dict.fromkeys(field_matches))
            return f"{field_text} ({', '.join(unique_en)})"
        return field_text

    expanded_assessment = expand_field(raw_assessment)
    expanded_subjective = expand_field(raw_subjective)
    expanded_objective = expand_field(raw_objective)

    return {
        "assessment": expanded_assessment,
        "subjective": expanded_subjective,
        "objective": expanded_objective,
        "plan": raw_plan,
        "raw_assessment": raw_assessment,
        "raw_subjective": raw_subjective,
        "raw_objective": raw_objective,
        "clinical_concepts_en": list(dict.fromkeys(matched_concepts)),
        "detected_language": "id" if is_indonesian else "en"
    }


def get_candidate_criteria(soap_dict, categories, subcodes, max_options=30):
    """
    Select relevant candidate options for Jev's Choice question.
    Jev Choice supports up to 255 options. To maximize focus and avoid dilution,
    we retrieve the most contextually relevant categories + specific codes.
    Supports both bilingual normalized state and raw SOAP input.
    """
    # Combine text from SOAP (checking both expanded and raw fields)
    full_text = " ".join([
        flatten_soap_field(soap_dict.get("assessment", "")),
        flatten_soap_field(soap_dict.get("objective", "")),
        flatten_soap_field(soap_dict.get("subjective", ""))
    ]).lower()

    # Pre-select based on clinical keywords & categories
    scored_candidates = []
    
    # Check all categories (74 in table)
    for code, desc in categories.items():
        score = 0
        desc_lower = desc.lower()
        words = re.findall(r"\w+", desc_lower)
        for w in words:
            if len(w) > 3 and w in full_text:
                score += 2
        # Extra points for direct code or exact term match
        if code.lower() in full_text:
            score += 5
        scored_candidates.append((score, code, desc))

    # Also check specific prominent subcodes
    for code, desc in subcodes.items():
        score = 0
        desc_lower = desc.lower()
        words = re.findall(r"\w+", desc_lower)
        for w in words:
            if len(w) > 3 and w in full_text:
                score += 3
        if score > 0:
            scored_candidates.append((score, code, desc))

    # Sort candidates by keyword relevance
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Take top candidates (up to max_options)
    criteria = {}
    for _, code, desc in scored_candidates[:max_options]:
        criteria[code] = desc

    # Always ensure staple categories are available as safety options
    staple_fallbacks = {
        "I50": "Heart failure (congestive heart failure)",
        "I20": "Angina pectoris",
        "I21": "Acute myocardial infarction",
        "I30": "Acute pericarditis",
        "I10": "Essential (primary) hypertension",
        "R07": "Pain in throat and chest (precordial pain, chest pain unspecified)",
        "OTHER": "Other condition or symptom without definitive heart disease match"
    }
    for code, desc in staple_fallbacks.items():
        if code not in criteria and len(criteria) < 40:
            criteria[code] = desc

    return criteria


def mock_jev_choice(state, criteria):
    """
    Simulate Jev's System One Choice primitive locally.
    Produces a calibrated softmax probability distribution and confidence score.
    Confidence formula from Paper 1: (K * p_max - 1) / (K - 1)
    """
    # Combine state for scoring (using flattened strings)
    assessment_text = flatten_soap_field(state.get("assessment", "")).lower()
    objective_text = flatten_soap_field(state.get("objective", "")).lower()
    subjective_text = flatten_soap_field(state.get("subjective", "")).lower()

    raw_scores = {}
    for code, desc in criteria.items():
        desc_words = [w for w in re.findall(r"\w+", desc.lower()) if len(w) > 3]
        score = 0.5  # Base smoothing prior

        # Precision/coverage: favor specific direct diagnoses over lengthy complication clauses
        matched_ass = [w for w in desc_words if w in assessment_text]
        coverage = (len(matched_ass) / len(desc_words)) if desc_words else 0.0

        # 1. Assessment section has highest weight (x5), boosted by coverage
        if matched_ass:
            score += sum(5.0 for _ in matched_ass) * (0.4 + 0.6 * coverage)

        # 2. Objective section (x2)
        for w in desc_words:
            if w in objective_text:
                score += 2.0

        # 3. Subjective section (x1)
        for w in desc_words:
            if w in subjective_text:
                score += 1.0

        # Penalize OTHER unless all scores are very low
        if code == "OTHER":
            score = 1.0

        raw_scores[code] = score

    # Softmax conversion to get calibrated probabilities
    max_raw = max(raw_scores.values()) if raw_scores else 1.0
    exp_scores = {k: math.exp(min((v - max_raw) / 1.5, 20.0)) for k, v in raw_scores.items()}
    total_exp = sum(exp_scores.values())
    probabilities = {k: round(v / total_exp, 4) for k, v in exp_scores.items()}

    # Selected choice is highest probability
    best_choice = max(probabilities, key=probabilities.get)
    p_max = probabilities[best_choice]
    k_options = len(criteria)

    # Jev confidence score formula: (K * p_max - 1) / (K - 1)
    if k_options > 1:
        confidence = max(0.0, (k_options * p_max - 1.0) / (k_options - 1.0))
    else:
        confidence = p_max

    # Heuristic adjustment: If Assessment was completely blank/empty, penalize confidence
    raw_assessment_check = state.get("raw_assessment") if "raw_assessment" in state else assessment_text
    if not str(raw_assessment_check).strip() or str(raw_assessment_check) in ("none", "[]", "{}"):
        confidence = min(confidence, 0.35)

    return {
        "choice": best_choice,
        "confidence": round(confidence, 3),
        "probabilities": probabilities
    }


def call_live_jev_api(api_key, state, criteria):
    """Call TypeSafe AI Jev System One endpoint."""
    import urllib.request
    import urllib.error

    payload = {
        "model": JEV_MODEL,
        "state": state,
        "questions": {
            "icd10_diagnosis": {
                "type": "choice",
                "instructions": (
                    "Select the primary ICD-10 cardiovascular disease code or category "
                    "that most accurately classifies the physician's diagnosis and clinical findings in the state."
                ),
                "criteria": criteria
            }
        }
    }

    req = urllib.request.Request(
        JEV_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            answer = data.get("answers", {}).get("icd10_diagnosis", {})
            return {
                "choice": answer.get("choice"),
                "confidence": answer.get("confidence", 0.0),
                "probabilities": answer.get("probabilities", {})
            }
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        raise RuntimeError(f"Jev API HTTP {e.code}: {err_msg}")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Jev API: {e}")


def map_soap_to_icd10(soap_data, force_mock=False):
    """
    Main function to map a SOAP dictionary/object to an ICD-10 decision.
    Automatically handles both Indonesian and English SOAP notes through
    clinical concept normalization.
    
    Returns:
        dict: {
            "code": str,
            "description": str,
            "confidence": float,
            "status": "AUTO_ACCEPT" | "FLAG_FOR_REVIEW",
            "mode": "LIVE_API" | "MOCK_SIMULATION",
            "detected_language": "id" | "en",
            "clinical_concepts": list,
            "top_probabilities": dict
        }
    """
    # 1. Normalize and extract canonical bilingual clinical concepts
    normalized_state = normalize_and_expand_bilingual_soap(soap_data)

    # 2. Retrieve targeted ICD-10 candidates from the 367 PERKI cardiovascular table
    categories, subcodes = load_cardiovascular_table()
    criteria = get_candidate_criteria(normalized_state, categories, subcodes)

    # 3. Determine execution mode (Live Jev System One vs Mock Simulation)
    api_key = None if force_mock else load_api_key()
    mode = "LIVE_API" if api_key else "MOCK_SIMULATION"

    # Prepare structured state for Jev
    jev_state = {
        "assessment": normalized_state["assessment"],
        "subjective": normalized_state["subjective"],
        "objective": normalized_state["objective"],
        "detected_language": normalized_state["detected_language"],
        "clinical_concepts": normalized_state["clinical_concepts_en"]
    }

    if mode == "LIVE_API":
        try:
            result = call_live_jev_api(api_key, jev_state, criteria)
        except Exception as e:
            print(f"[Warning] Live API call failed ({e}). Falling back to Mock Simulation.")
            result = mock_jev_choice(normalized_state, criteria)
            mode = "MOCK_SIMULATION (Fallback)"
    else:
        result = mock_jev_choice(normalized_state, criteria)

    choice_code = result.get("choice", "OTHER")
    conf = float(result.get("confidence", 0.0))

    # Lookup code description
    all_codes = {**categories, **subcodes}
    all_codes["OTHER"] = "Other condition / unconfirmed diagnosis"
    all_codes["R07"] = "Precordial / chest pain unspecified"
    code_desc = all_codes.get(choice_code, criteria.get(choice_code, "Unspecified"))

    # Confidence-gating check (Paper 1 selective prediction threshold)
    status = "AUTO_ACCEPT" if conf >= CONFIDENCE_THRESHOLD else "FLAG_FOR_REVIEW"

    # Sort top 5 probabilities for visibility
    probs = result.get("probabilities", {})
    sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5])

    return {
        "code": choice_code,
        "description": code_desc,
        "confidence": conf,
        "status": status,
        "mode": mode,
        "detected_language": normalized_state.get("detected_language", "en"),
        "clinical_concepts": normalized_state.get("clinical_concepts_en", []),
        "top_probabilities": sorted_probs
    }


# ===========================================================================
# Clinical Case Extractors for CAR0001 - CAR0005
# ===========================================================================

def build_car_case_soap(case_id):
    """Construct representative SOAP dictionary for CAR0001-CAR0005 based on dialogue."""
    cases = {
        "CAR0001": {
            "subjective": "39yo male reports sharp left-sided chest pain lasting 8h, worse when lying flat. Heavy smoker (15 pack-years), cannabis use. Father had heart attack at age 45.",
            "objective": "No physical exam documented in audio dialogue (anamnesis only).",
            "assessment": "",  # Empty - consultation cut off after history
            "plan": ""
        },
        "CAR0002": {
            "subjective": "Male reports acute onset dull chest pressure (7/10) started 30 mins ago with shortness of breath, worse walking and lying down. Heavy smoker, past crystal meth use 6 days ago.",
            "objective": "Audio recording ended mid-interview during social history.",
            "assessment": "",  # Empty
            "plan": ""
        },
        "CAR0003": {
            "subjective": "Patient reports 2 months of progressive exertional dyspnea (>50m) and severe orthopnea requiring stacked pillows. Cough with frothy sputum. Bilateral leg swelling. Past history of MI with stenting and hypercholesterolemia. Former 40 pack-year smoker.",
            "objective": "Bilateral lower extremity edema (leg swelling) present for 2 months. Orthopnea on supine positioning.",
            "assessment": "Congestive heart failure (likely post-infarction cardiomyopathy and fluid overload).",
            "plan": "Order diagnostic investigations and initiate diuretic therapy with Lasix to relieve fluid buildup."
        },
        "CAR0004": {
            "subjective": "Young male university student with sharp chest pain for 3 days, lasts seconds, worse with coughing and lying down, relieved by leaning forward. Radiates to neck.",
            "objective": "Normal vitals discussed, awaiting investigations.",
            "assessment": "Rule-out acute pericarditis vs musculoskeletal chest pain.",
            "plan": "Order ECG to evaluate heart function and blood work."
        },
        "CAR0005": {
            "subjective": "Female with left chest sharp discomfort lasting 20-30 mins per episode over 2 weeks, non-exertional (occurs when driving/working), high divorce stress. Normal exercise tolerance.",
            "objective": "No acute distress, no fever.",
            "assessment": "Non-cardiac / atypical chest pain secondary to stress.",
            "plan": "Refer to family doctor for follow-up and next steps."
        }
    }
    return cases.get(case_id)


def run_car_evaluation(force_mock=False):
    """Run Jev mapper across CAR0001–CAR0005 and print formatted table."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = []

    print("\n" + "=" * 80)
    print(" JEV SYSTEM ONE ICD-10 MAPPER: EVALUATION ON CAR0001 - CAR0005")
    print("=" * 80)

    for case_id in ["CAR0001", "CAR0002", "CAR0003", "CAR0004", "CAR0005"]:
        soap = build_car_case_soap(case_id)
        res = map_soap_to_icd10(soap, force_mock=force_mock)
        res["case_id"] = case_id
        results.append(res)

        print(f"\n> Case: {case_id}")
        print(f"  Mode:            {res['mode']}")
        print(f"  Predicted Code:  {res['code']} ({res['description']})")
        print(f"  Confidence:      {res['confidence']:.3f}")
        print(f"  Decision Status: {res['status']}")
        print(f"  Top Probabilities: {res['top_probabilities']}")

    # Save summary CSV
    out_csv = OUTPUT_DIR / "car_cases_jev_mapping.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "predicted_icd10", "description", "confidence", "status", "mode", "top_candidates"])
        for r in results:
            writer.writerow([
                r["case_id"],
                r["code"],
                r["description"],
                r["confidence"],
                r["status"],
                r["mode"],
                json.dumps(r["top_probabilities"])
            ])

    print("\n" + "=" * 80)
    print(f" Results successfully saved to: {out_csv}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Jev System One SOAP to ICD-10 Mapper")
    parser.add_argument("--test-car-cases", action="store_true", help="Run evaluation on CAR0001–CAR0005")
    parser.add_argument("--mock", action="store_true", help="Force mock / dry-run mode (no API key needed)")
    parser.add_argument("--case", type=str, help="Run single case (e.g. CAR0003)")
    parser.add_argument("--soap-file", type=str, help="Path to JSON file containing SOAP data")
    args = parser.parse_args()

    if args.test_car_cases or (not args.case and not args.soap_file):
        run_car_evaluation(force_mock=args.mock)
    elif args.case:
        soap = build_car_case_soap(args.case.upper())
        if not soap:
            print(f"Error: Unknown case {args.case}")
            return
        res = map_soap_to_icd10(soap, force_mock=args.mock)
        print(json.dumps(res, indent=2))
    elif args.soap_file:
        with open(args.soap_file, encoding="utf-8") as f:
            soap = json.load(f)
        res = map_soap_to_icd10(soap, force_mock=args.mock)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
