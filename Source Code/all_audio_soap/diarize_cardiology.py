"""
Re-diarize CAR0002-CAR0005 (Doctor / Patient) from their Whisper segments.
Originals (generic "Speaker" labels) are backed up as *_transcript.speaker_backup.txt.
Only the transcript .txt is rewritten; SOAP CSVs are NOT touched.
"""
import json, os, shutil, sys, time
from pathlib import Path

SOURCE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SOURCE_DIR))

from stt_pipeline_gpu import diarize_and_extract_soap

AUDIO_DIR = SOURCE_DIR.parent / 'audio_sample' / 'audio_recordings' / 'Audio_Recordings'
CASES = ['CAR0002', 'CAR0003', 'CAR0004', 'CAR0005']

for case_id in CASES:
    seg_path = AUDIO_DIR / f'{case_id}_segments.json'
    out_path = AUDIO_DIR / f'{case_id}_transcript.txt'
    backup = AUDIO_DIR / f'{case_id}_transcript.speaker_backup.txt'

    segments = json.loads(seg_path.read_text(encoding='utf-8'))
    result = diarize_and_extract_soap(segments)
    if result is None:
        print(f'FAILED {case_id}')
        continue
    segs = result.get('diarized_segments', [])
    speakers = {s.get('speaker') for s in segs}
    if not segs or not speakers & {'Doctor', 'Patient'}:
        print(f'SKIP {case_id}: no Doctor/Patient labels returned')
        continue

    if out_path.exists() and not backup.exists():
        shutil.copy(out_path, backup)
    lines = [f'[{s.get("start", 0):.1f}s->{s.get("end", 0):.1f}s] '
             f'{s.get("speaker", "?")}: {s.get("text", "").strip()}' for s in segs]
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'OK {case_id}: {len(segs)} segments, speakers={speakers}')
    time.sleep(3)
