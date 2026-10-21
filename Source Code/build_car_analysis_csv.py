"""
Build analysis_CAR_cases_only.csv from existing predicted/ground truth SOAP files.

Columns:
  - case_number
  - whisper_transcription_dialogue
  - ground_truth_dialogue
  - predicted_soap_analysis (from Whisper transcription)
  - ground_truth_soap_analysis
"""

import csv
import os

BASE = os.path.dirname(os.path.abspath(__file__))

PREDICTED_DIR = os.path.join(BASE, "all_audio_soap", "predicted")
GT_DIR = os.path.join(BASE, "all_audio_soap", "ground_truth")
AUDIO_REC_DIR = os.path.join(BASE, "..", "audio_sample", "audio_recordings", "Audio_Recordings")
CLEAN_TRANS_DIR = os.path.join(BASE, "..", "audio_sample", "audio_recordings", "Clean_Transcripts")

OUTPUT_CSV = os.path.join(BASE, "analysis_CAR_cases_only.csv")

CAR_CASES = ["CAR0001", "CAR0002", "CAR0003", "CAR0004", "CAR0005"]

SOAP_FIELDS = [
    "audio_file", "summary", "encounter_type", "additional_notes", "extraction_confidence",
    "subjective.chief_complaint", "subjective.symptoms", "subjective.symptom_onset",
    "subjective.medical_history", "subjective.current_medications", "subjective.allergies",
    "objective.physical_exam_findings", "objective.vital_signs",
    "assessment.diagnosis", "assessment.differential_diagnosis",
    "plan.prescribed_medications", "plan.treatment_plan", "plan.investigations_ordered",
    "plan.referrals", "plan.follow_up"
]


def read_soap_csv(filepath):
    """Read a SOAP CSV file and return the data row as a dict."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    if rows:
        return rows[0]
    return None


def soap_dict_to_str(soap_dict):
    """Convert the SOAP dict to a readable multi-line string."""
    if soap_dict is None:
        return ""
    parts = []
    for field in SOAP_FIELDS:
        val = soap_dict.get(field, "").strip()
        if val:
            parts.append(f"{field}: {val}")
    return "\n".join(parts)


def read_text_file(filepath):
    """Read a plain text file. Return empty string if not found."""
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    rows = []

    for case in CAR_CASES:
        print(f"Processing {case}...")

        # --- Predicted SOAP ---
        predicted_path = os.path.join(PREDICTED_DIR, f"{case}_soap.csv")
        predicted_soap = read_soap_csv(predicted_path)
        if predicted_soap is None:
            print(f"  WARNING: Predicted SOAP not found: {predicted_path}")

        # --- Ground Truth SOAP ---
        gt_soap_path = os.path.join(GT_DIR, f"{case}_soap.csv")
        gt_soap = read_soap_csv(gt_soap_path)
        if gt_soap is None:
            print(f"  WARNING: Ground truth SOAP not found: {gt_soap_path}")

        # --- Whisper Transcription Dialogue ---
        whisper_path = os.path.join(AUDIO_REC_DIR, f"{case}_transcript.txt")
        whisper_dialogue = read_text_file(whisper_path)
        if not whisper_dialogue:
            print(f"  INFO: Whisper transcript not found (will be empty): {whisper_path}")

        # --- Ground Truth Dialogue ---
        gt_dialogue_path = os.path.join(CLEAN_TRANS_DIR, f"{case}.txt")
        gt_dialogue = read_text_file(gt_dialogue_path)
        if not gt_dialogue:
            print(f"  WARNING: Ground truth dialogue not found: {gt_dialogue_path}")

        rows.append({
            "case_number": case,
            "whisper_transcription_dialogue": whisper_dialogue,
            "ground_truth_dialogue": gt_dialogue,
            "predicted_soap_analysis": soap_dict_to_str(predicted_soap),
            "ground_truth_soap_analysis": soap_dict_to_str(gt_soap),
        })

    # Write output CSV
    fieldnames = [
        "case_number",
        "whisper_transcription_dialogue",
        "ground_truth_dialogue",
        "predicted_soap_analysis",
        "ground_truth_soap_analysis",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Output saved to:\n  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
