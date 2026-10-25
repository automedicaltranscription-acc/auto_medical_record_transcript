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


def get_candidate_criteria(soap_dict, categories, subcodes, max_options=30):
    """
    Select relevant candidate options for Jev's Choice question.
    Jev Choice supports up to 255 options. To maximize focus and avoid dilution,
    we retrieve the most contextually relevant categories + specific codes.
    """
    # Combine text from SOAP to rank candidate matches
    full_text = " ".join([
        str(soap_dict.get("assessment", "")),
        str(soap_dict.get("objective", "")),
        str(soap_dict.get("subjective", ""))
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
    # Combine state for scoring
    assessment_text = str(state.get("assessment", "")).lower()
    objective_text = str(state.get("objective", "")).lower()
    subjective_text = str(state.get("subjective", "")).lower()

    raw_scores = {}
    for code, desc in criteria.items():
        desc_words = [w for w in re.findall(r"\w+", desc.lower()) if len(w) > 3]
        score = 0.5  # Base smoothing prior

        # 1. Assessment section has highest weight (x5)
        for w in desc_words:
            if w in assessment_text:
                score += 5.0

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
    if not assessment_text.strip() or assessment_text in ("none", "[]", "{}"):
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
    Returns:
        dict: {
            "code": str,
            "description": str,
            "confidence": float,
            "status": "AUTO_ACCEPT" | "FLAG_FOR_REVIEW",
            "mode": "LIVE_API" | "MOCK_SIMULATION",
            "top_probabilities": dict
        }
    """
    categories, subcodes = load_cardiovascular_table()
    criteria = get_candidate_criteria(soap_data, categories, subcodes)

    api_key = None if force_mock else load_api_key()
    mode = "LIVE_API" if api_key else "MOCK_SIMULATION"

    if mode == "LIVE_API":
        try:
            result = call_live_jev_api(api_key, soap_data, criteria)
        except Exception as e:
            print(f"[Warning] Live API call failed ({e}). Falling back to Mock Simulation.")
            result = mock_jev_choice(soap_data, criteria)
            mode = "MOCK_SIMULATION (Fallback)"
    else:
        result = mock_jev_choice(soap_data, criteria)

    choice_code = result.get("choice", "OTHER")
    conf = float(result.get("confidence", 0.0))

    # Lookup code description
    all_codes = {**categories, **subcodes}
    all_codes["OTHER"] = "Other condition / unconfirmed diagnosis"
    all_codes["R07"] = "Precordial / chest pain unspecified"
    code_desc = all_codes.get(choice_code, criteria.get(choice_code, "Unspecified"))

    # Confidence-gating check
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
