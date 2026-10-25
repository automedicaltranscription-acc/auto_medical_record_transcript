"""
extract_icd10_table.py
======================
Extracts the "Daftar Penyakit Kardiovaskular berdasarkan ICD-10" table from
'PPK Kardiologi.pdf'.

The PDF has no extractable text layer on these pages, so each page is rendered
to an image (PyMuPDF) and read with RapidOCR. Lines of the form
    (I01.0) Acute rheumatic pericarditis
are parsed, OCR letter/digit confusions in the code are repaired, and every row
is validated. Rows that look doubtful are flagged with needs_review=True so they
can be checked by hand against the PDF.

Outputs (in the same folder as this script):
  icd10_cardiovascular.csv   parsed table (code, description, level, parent, ...)
  icd10_ocr_raw.csv          raw OCR lines (for auditing)

Usage:
  python extract_icd10_table.py
  python extract_icd10_table.py --first 364 --last 378      # 1-based PDF pages
"""

import argparse
import csv
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pymupdf
from rapidocr_onnxruntime import RapidOCR

SOURCE_DIR = Path(__file__).parent
PDF_PATH = SOURCE_DIR.parent / "PPK Kardiologi.pdf"
OUT_DIR = SOURCE_DIR

LOW_CONF = 0.80
# One code token: optional letter-like char, two digits, optional .digits
CODE_TOKEN = r"[A-Za-z1|]?\d{2}(?:\.\d+)?"
LINE_RE = re.compile(
    rf"^\(\s*({CODE_TOKEN})\s*(?:[-\u2013\u2014]\s*({CODE_TOKEN}))?\s*\)\s*(.*)$"
)
TOKEN_RE = re.compile(r"^([A-Za-z1|]?)(\d{2})(\.\d+)?$")
SKIP_MARKERS = ("Panduan Praktik Klinis", "Daftar Penyakit")


def normalize_code(tok):
    """Repair OCR letter confusions. Returns (code, was_fixed) or (None, False)."""
    tok = tok.replace(" ", "")
    m = TOKEN_RE.match(tok)
    if not m:
        return None, False
    letter, digits, frac = m.group(1), m.group(2), m.group(3) or ""
    if letter in ("", "1", "l", "|", "i", "I"):
        fixed = letter != "I"
        letter = "I"  # cardiovascular chapter; a missing/odd leading char is an 'I'
    else:
        fixed = letter != letter.upper()
        letter = letter.upper()
    return f"{letter}{digits}{frac}", fixed


def ocr_page(ocr, page, dpi=200):
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    result, _ = ocr(img)
    lines = []
    for box, text, conf in result or []:
        lines.append({"y": float(box[0][1]), "x": float(box[0][0]),
                      "conf": float(conf), "text": text.strip()})
    lines.sort(key=lambda r: (round(r["y"] / 8), r["x"]))
    return lines


def parse(pages_lines):
    rows, block_range = [], None
    for page_no, lines in pages_lines:
        for ln in lines:
            text = ln["text"]
            if any(mk in text for mk in SKIP_MARKERS):
                continue
            m = LINE_RE.match(text)
            if not m:
                # continuation of a wrapped description
                if rows and rows[-1]["page"] == page_no:
                    rows[-1]["description"] += " " + text
                    rows[-1]["min_conf"] = min(rows[-1]["min_conf"], ln["conf"])
                elif text:
                    rows.append({"page": page_no, "code": "", "description": text,
                                 "level": "unparsed", "parent": "", "ocr_fixed": False,
                                 "min_conf": ln["conf"], "raw": text})
                continue
            tok1, tok2, desc = m.group(1), m.group(2), m.group(3).strip()
            code, fixed = normalize_code(tok1)
            code2, fixed2 = normalize_code(tok2) if tok2 else (None, False)
            if code is None or (tok2 and code2 is None):
                rows.append({"page": page_no, "code": "", "description": text,
                             "level": "unparsed", "parent": "", "ocr_fixed": False,
                             "min_conf": ln["conf"], "raw": text})
                continue
            if tok2:
                level, code_out, block_range = "block", f"{code}-{code2}", (code, code2)
                parent = ""
            elif "." in code:
                level, code_out, parent = "code", code, code.split(".")[0]
            else:
                level, code_out = "category", code
                parent = f"{block_range[0]}-{block_range[1]}" if block_range else ""
            rows.append({"page": page_no, "code": code_out, "description": desc,
                         "level": level, "parent": parent,
                         "ocr_fixed": fixed or fixed2, "min_conf": ln["conf"], "raw": text})
    return rows


def validate(rows):
    seen_cats, prev_key = set(), None
    for r in rows:
        flags = []
        if r["level"] == "unparsed":
            flags.append("unparsed line")
        if r["min_conf"] < LOW_CONF:
            flags.append("low OCR confidence")
        if r["level"] == "category":
            seen_cats.add(r["code"])
            lo, hi = (r["parent"].split("-") + [""])[:2]
            if lo and hi and not (lo <= r["code"] <= hi):
                flags.append("category outside block range")
        if r["level"] == "code" and r["parent"] not in seen_cats:
            flags.append("parent category not seen")
        if r["level"] in ("category", "code"):
            if prev_key and r["code"] < prev_key:
                flags.append("code out of order")
            prev_key = r["code"]
        if not r["description"]:
            flags.append("empty description")
        r["needs_review"] = bool(flags)
        r["review_reason"] = "; ".join(flags)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=364, help="first PDF page (1-based)")
    ap.add_argument("--last", type=int, default=378, help="last PDF page (1-based)")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    doc = pymupdf.open(PDF_PATH)
    ocr = RapidOCR()

    pages_lines = []
    for p in range(args.first, args.last + 1):
        lines = ocr_page(ocr, doc[p - 1])
        pages_lines.append((p, lines))
        print(f"page {p}: {len(lines)} OCR lines")

    with open(OUT_DIR / "icd10_ocr_raw.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["page", "y", "x", "conf", "text"])
        for p, lines in pages_lines:
            for ln in lines:
                w.writerow([p, round(ln["y"]), round(ln["x"]), round(ln["conf"], 3), ln["text"]])

    rows = validate(parse(pages_lines))
    cols = ["code", "description", "level", "parent", "page", "ocr_fixed",
            "min_conf", "needs_review", "review_reason", "raw"]
    with open(OUT_DIR / "icd10_cardiovascular.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r["min_conf"] = round(r["min_conf"], 3)
            w.writerow(r)

    n = lambda lv: sum(r["level"] == lv for r in rows)
    print(f"\nrows: {len(rows)}  blocks: {n('block')}  categories: {n('category')}  "
          f"codes: {n('code')}  unparsed: {n('unparsed')}")
    print(f"needs review: {sum(r['needs_review'] for r in rows)}  "
          f"(OCR letter repaired in {sum(r['ocr_fixed'] for r in rows)} rows)")
    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
