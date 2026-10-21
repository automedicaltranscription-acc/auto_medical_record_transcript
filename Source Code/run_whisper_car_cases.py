"""
run_whisper_car_cases.py
========================
Runs Whisper (faster-whisper, GPU float16) on CAR0002–CAR0005, one case at a time.
After each case:
  1. Saves the raw Whisper transcript → Audio_Recordings/CAR000X_transcript.txt
  2. Updates analysis_CAR_cases_only.csv with the new whisper_transcription_dialogue

Flow per case:
  transcribe → save _transcript.txt → update CSV → next case

If interrupted mid-way, already-saved transcripts are preserved.
Cases with existing _transcript.txt are skipped (resume-safe).
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SOURCE_DIR      = Path(__file__).parent
AUDIO_DIR       = SOURCE_DIR.parent / 'audio_sample' / 'audio_recordings' / 'Audio_Recordings'
CLEAN_TRANS_DIR = SOURCE_DIR.parent / 'audio_sample' / 'audio_recordings' / 'Clean_Transcripts'
PREDICTED_DIR   = SOURCE_DIR / 'all_audio_soap' / 'predicted'
GT_DIR          = SOURCE_DIR / 'all_audio_soap' / 'ground_truth'
OUTPUT_CSV      = SOURCE_DIR / 'analysis_CAR_cases_only.csv'

# CAR cases to process (CAR0001 already has a transcript)
CASES = ['CAR0002', 'CAR0003', 'CAR0004', 'CAR0005']

# ---------------------------------------------------------------------------
# CUDA library preload (same as stt_pipeline_gpu.py)
# ---------------------------------------------------------------------------
import ctypes
if sys.platform != 'win32':
    _cuda_lib_dir = os.path.join(
        os.path.dirname(sys.executable), '..', 'lib',
        f'python{sys.version_info.major}.{sys.version_info.minor}',
        'site-packages', 'nvidia', 'cu13', 'lib'
    )
    if os.path.exists(_cuda_lib_dir):
        for _lib in ['libcublasLt.so.12', 'libcublas.so.12', 'libcudart.so.12', 'libnvrtc.so.12']:
            _lib_path = os.path.join(_cuda_lib_dir, _lib)
            if os.path.exists(_lib_path):
                try:
                    ctypes.CDLL(_lib_path, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass

# ---------------------------------------------------------------------------
# Lazy Whisper model loader (loaded once, reused)
# ---------------------------------------------------------------------------
_whisper_model = None

def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import torch
        from faster_whisper import WhisperModel
        device      = 'cuda' if torch.cuda.is_available() else 'cpu'
        compute     = 'float16' if device == 'cuda' else 'int8'
        print(f'\n[Whisper] Loading medium model on {device} ({compute})...')
        t0 = time.time()
        _whisper_model = WhisperModel('medium', device=device, compute_type=compute)
        print(f'[Whisper] Loaded in {time.time()-t0:.1f}s')
    return _whisper_model


# ---------------------------------------------------------------------------
# Transcribe one audio file → list of segment dicts
# ---------------------------------------------------------------------------
MEDICAL_INITIAL_PROMPT = (
    'Doctor patient consultation. Medications: Lisinopril, Metformin, Rosuvastatin, '
    'Atorvastatin, Amlodipine, Omeprazole, Metoprolol, Amoxicillin, Azithromycin, '
    'Sertraline, Gabapentin, Hydrochlorothiazide, Losartan, Prednisone, Albuterol, '
    'Levothyroxine, Furosemide, Pantoprazole, Clopidogrel, Warfarin. '
    'Symptoms: hypertension, diabetes, dyspnea, tachycardia, bradycardia, '
    'myocardial infarction, pericarditis, pneumothorax. '
    'Dosage: milligrams, micrograms, twice daily, once daily, as needed.'
)

def transcribe(audio_path: Path) -> list[dict]:
    """
    Run faster-whisper on audio_path. Loads from _segments.json cache if available.
    Returns list of {id, start, end, text} dicts.
    """
    cache_path = audio_path.with_suffix('') .parent / (audio_path.stem + '_segments.json')

    if cache_path.exists():
        print(f'  [Cache] Loading existing segments from {cache_path.name}')
        with open(cache_path, encoding='utf-8') as f:
            segments = json.load(f)
        print(f'  [Cache] {len(segments)} segments loaded')
        return segments

    model = _get_whisper_model()
    print(f'  [Whisper] Transcribing {audio_path.name}...')
    t0 = time.time()
    segments_gen, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        initial_prompt=MEDICAL_INITIAL_PROMPT,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    segments = [
        {'id': i, 'start': seg.start, 'end': seg.end, 'text': seg.text.strip()}
        for i, seg in enumerate(segments_gen)
    ]
    elapsed = time.time() - t0
    print(f'  [Whisper] Language: {info.language} | Segments: {len(segments)} | Time: {elapsed:.1f}s')

    # Save cache
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2)
    print(f'  [Whisper] Segments cached → {cache_path.name}')
    return segments


# ---------------------------------------------------------------------------
# Build transcript text from raw Whisper segments (no speaker labels)
# ---------------------------------------------------------------------------
def segments_to_transcript(segments: list[dict]) -> str:
    """
    Format segments the same way as CAR0001_transcript.txt:
      [0.0s->8.9s] Patient: <text>
    Since we are NOT running GPT for diarization (no Gemini/GPT tokens),
    the speaker label is left as 'Speaker' — it's honest about what we know.
    """
    lines = []
    for seg in segments:
        start = seg.get('start', 0.0)
        end   = seg.get('end', 0.0)
        text  = seg.get('text', '').strip()
        lines.append(f'[{start:.1f}s->{end:.1f}s] Speaker: {text}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Read SOAP CSV helper
# ---------------------------------------------------------------------------
SOAP_FIELDS = [
    'audio_file', 'summary', 'encounter_type', 'additional_notes', 'extraction_confidence',
    'subjective.chief_complaint', 'subjective.symptoms', 'subjective.symptom_onset',
    'subjective.medical_history', 'subjective.current_medications', 'subjective.allergies',
    'objective.physical_exam_findings', 'objective.vital_signs',
    'assessment.diagnosis', 'assessment.differential_diagnosis',
    'plan.prescribed_medications', 'plan.treatment_plan', 'plan.investigations_ordered',
    'plan.referrals', 'plan.follow_up'
]

def read_soap_csv(filepath: Path) -> dict | None:
    if not filepath.exists():
        return None
    with open(filepath, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None

def soap_dict_to_str(soap_dict: dict | None) -> str:
    if soap_dict is None:
        return ''
    parts = []
    for field in SOAP_FIELDS:
        val = soap_dict.get(field, '').strip()
        if val:
            parts.append(f'{field}: {val}')
    return '\n'.join(parts)

def read_text(filepath: Path) -> str:
    if not filepath.exists():
        return ''
    return filepath.read_text(encoding='utf-8').strip()


# ---------------------------------------------------------------------------
# Rebuild full CSV from scratch (so it stays consistent after each update)
# ---------------------------------------------------------------------------
ALL_CASES = ['CAR0001', 'CAR0002', 'CAR0003', 'CAR0004', 'CAR0005']
FIELDNAMES = [
    'case_number',
    'whisper_transcription_dialogue',
    'ground_truth_dialogue',
    'predicted_soap_analysis',
    'ground_truth_soap_analysis',
]

def rebuild_csv():
    rows = []
    for case in ALL_CASES:
        whisper_path = AUDIO_DIR / f'{case}_transcript.txt'
        gt_dial_path = CLEAN_TRANS_DIR / f'{case}.txt'
        pred_soap_path = PREDICTED_DIR / f'{case}_soap.csv'
        gt_soap_path   = GT_DIR / f'{case}_soap.csv'

        rows.append({
            'case_number':                  case,
            'whisper_transcription_dialogue': read_text(whisper_path),
            'ground_truth_dialogue':          read_text(gt_dial_path),
            'predicted_soap_analysis':        soap_dict_to_str(read_soap_csv(pred_soap_path)),
            'ground_truth_soap_analysis':     soap_dict_to_str(read_soap_csv(gt_soap_path)),
        })

    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    print(f'  [CSV] Rebuilt → {OUTPUT_CSV.name}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print('=' * 60)
    print('  Whisper Transcription — CAR0002 to CAR0005')
    print('  (Transcript saved per case before moving to next)')
    print('=' * 60)

    for case in CASES:
        print(f'\n{"─"*50}')
        print(f'  Case: {case}')
        print(f'{"─"*50}')

        audio_path      = AUDIO_DIR / f'{case}.mp3'
        transcript_path = AUDIO_DIR / f'{case}_transcript.txt'

        # ── Skip if transcript already exists ──────────────────────────
        if transcript_path.exists():
            print(f'  [SKIP] Transcript already exists: {transcript_path.name}')
            print(f'  (Delete it to re-run)')
            continue

        if not audio_path.exists():
            print(f'  [ERROR] Audio file not found: {audio_path}')
            continue

        # ── Step 1: Transcribe ──────────────────────────────────────────
        try:
            segments = transcribe(audio_path)
        except Exception as e:
            print(f'  [ERROR] Whisper failed for {case}: {e}')
            import traceback; traceback.print_exc()
            continue

        if not segments:
            print(f'  [ERROR] No segments returned for {case}')
            continue

        # ── Step 2: Save transcript .txt ────────────────────────────────
        transcript_text = segments_to_transcript(segments)
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript_text)
        print(f'  [SAVED] Transcript → {transcript_path.name}  ({len(segments)} segments)')

        # ── Step 3: Update the master CSV ──────────────────────────────
        rebuild_csv()
        print(f'  [DONE] {case} complete ✅')

    print(f'\n{"="*60}')
    print('  All done! Final CSV:')
    print(f'  {OUTPUT_CSV}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
