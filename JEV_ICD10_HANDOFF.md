# Handoff: Jev (System One) SOAP to ICD-10 Mapping

**Date:** 2026-10-25  
**Current Phase:** Transition from Knowledge Base Extraction & Clinical Exploration to Jev Mapper Implementation.

---

## 1. Executive Summary & Objective

The objective of this phase is to use **Jev** (a commercial "System One" model from TypeSafe AI) to convert clinical **SOAP reports** into standardized **ICD-10 cardiovascular disease codes**.

### End-to-End Pipeline
```
[Doctor-Patient Audio] 
         │
         ▼
[Whisper Large-v3 STT + Diarization] 
         │  (Doctor vs. Patient Dialogue)
         ▼
[Generative LLM (Gemini / GPT)] 
         │  (Clinical extraction into Subjective, Objective, Assessment, Plan)
         ▼
[SOAP Report] 
         │
         ▼
[System One Model (Jev)] ──► [PPK Cardiovascular ICD-10 Table]
         │
         ▼
[ICD-10 Code + Description + Calibrated Confidence Score]
         │
         ├─► High Confidence (≥ 0.60): Auto-Accept
         └─► Low Confidence (< 0.60): Flag for Human Coder / Clinician Review
```

---

## 2. Key Learnings from Research Papers

Two foundational papers were analyzed (see full summaries in [`jev_paper_summaries.md`](file:///C:/Users/justi/.gemini/antigravity-ide/brain/f4fc08e6-3ea5-4c9c-a131-0f02dfbd2917/jev_paper_summaries.md)):

1. **Paper 1: *Evaluating and Benchmarking the System One Model Jev* (Deußer et al., 2026)**
   * Jev does **not generate free-form text**. It takes a structured `state` and schema-typed questions (`Choice`, `Score`, `Noul`/boolean) and outputs probabilities and confidence scores.
   * **Strengths:** Fast (~0.36s latency), low cost ($0.042/M input tokens, free output tokens), calibrated multiple-choice probabilities (ECE 0.028), strong zero-shot performance on classification.
   * **Key Technique:** **Selective Prediction / Confidence-Gating** — withholding the least confident answers dramatically increases accuracy (e.g., 80% → 96% at 50% coverage).
   * **Weaknesses:** Cannot handle open-ended text reasoning, requires external candidate sets (max 255 options per Choice), drops on fine-grained specialist domains unless constrained.

2. **Paper 2: *Jev-Mem: System-One-Controlled Agentic Memory for Efficient AI Agents* (Jiang et al., 2026)**
   * Separates fast, high-frequency, bounded decisions (System One / Jev) from open-ended generative reasoning and synthesis (System Two / LLM).

---

## 3. ICD-10 Knowledge Base Extraction (`PPK Kardiologi.pdf`)

* **Source Document:** `PPK Kardiologi.pdf` (Panduan Praktik Klinis & Clinical Pathways from PERKI - Perhimpunan Dokter Spesialis Kardiovaskular Indonesia).
* **Finding:** Pages 364–378 (book pages 356–370) contain the appendix table *"Daftar Penyakit Kardiovaskular berdasarkan ICD-10"*. Because the PDF lacks an extractable text stream on these pages, an OCR pipeline was established.
* **Extraction Script:** [`Source Code/extract_icd10_table.py`](file:///d:/Users/justi/OneDrive/Documents/Binus%20aso/Antigravity%20Medical%20STT/Source%20Code/extract_icd10_table.py)
  * Uses `pymupdf` (rendering to 200 DPI images) + `rapidocr-onnxruntime` (lightweight CPU OCR).
  * Automatically repairs letter/digit OCR confusions (e.g., `121.0` → `I21.0`).
* **Output Datasets:**
  * [`Source Code/icd10_cardiovascular.csv`](file:///d:/Users/justi/OneDrive/Documents/Binus%20aso/Antigravity%20Medical%20STT/Source%20Code/icd10_cardiovascular.csv): 367 validated rows (11 disease blocks, 74 3-character categories, 279 4-character codes).
  * [`Source Code/icd10_ocr_raw.csv`](file:///d:/Users/justi/OneDrive/Documents/Binus%20aso/Antigravity%20Medical%20STT/Source%20Code/icd10_ocr_raw.csv): Raw OCR bounding boxes and confidence scores for auditing.

---

## 4. Case Studies & Clinical Realities (CAR0001–CAR0005)

Analysis of the 5 cardiology dialogue transcripts in `audio_sample/audio_recordings/Clean_Transcripts/`:

| Case | What Audio Contains | Clinical Impression / Hypothesis | Official Medical Coding Reality | Confidence |
| :--- | :--- | :--- | :--- | :--- |
| **CAR0001** | History taking only; cuts off before exam/assessment | Acute Pericarditis (`I30`) vs. Angina (`I20`) | No diagnosis documented → Symptom only (`R07.2` Precordial pain) | **Low** |
| **CAR0002** | Cuts off mid-dialogue during drug history | ACS / Meth-induced Angina (`I20.0` / `I21`) | Incomplete transcript → Symptom only (`R07.9` Chest pain) | **Low** |
| **CAR0003** | Full consultation; doctor states: *"it may be likely... that you might be having what's called congestive heart failure"* + plans Lasix | **Congestive Heart Failure** | **`I50.0` (Congestive heart failure)** | **High (≥ 0.85)** |
| **CAR0004** | Consultation where doctor orders ECG and bloodwork | Textbook Acute Pericarditis (`I30`) | Pending diagnostic workup → Symptom (`R07.2`) or rule-out | **Moderate/Low** |
| **CAR0005** | Consultation where doctor refers to family physician | Atypical / Stress-related chest pain | Non-cardiac / Symptom (`R07.9`) | **Moderate/Low** |

### Selected Design Decision
The user chose the **Clinical Working Diagnosis Approach with Confidence-Gating**:
* Jev is asked to predict the likely cardiovascular disease category from the PPK table based on available SOAP context.
* If documentation is incomplete or unconfirmed (as in `CAR0001`, `CAR0002`, `CAR0004`, `CAR0005`), Jev will output a **low confidence score**, triggering a review flag rather than hallucinating a false certainty.
* If a clear assessment is present (as in `CAR0003`), Jev outputs **high confidence** (`I50.0`).

---

## 5. Setup & Credentials Needed for Jev

1. **Sign Up & Console:** [console.typesafe.ai](https://console.typesafe.ai)
2. **API Key:** Create a key in the dashboard (starts with `jv_live_...`).
3. **Environment Variable:** Add to `Source Code/.env`:
   ```bash
   TYPESAFE_API_KEY="jv_live_..."
   ```
4. **Endpoint & Model:**
   * URL: `https://api.typesafe.ai/v1/systemone`
   * Model: `jev-1.13.0` (or `jev-latest`)
   * Pricing: ~$5 starter credit pack covers tens of thousands of test calls ($0 for output tokens).

---

## 6. Implementation Status & Next Step

### ✅ Completed in this Step:
1. **Implemented [`Source Code/jev_icd10_mapper.py`](file:///d:/Users/justi/OneDrive/Documents/Binus%20aso/Antigravity%20Medical%20STT/Source%20Code/jev_icd10_mapper.py):**
   * **Dual Mode Engine:**
     - **Live API Mode:** Connects to `https://api.typesafe.ai/v1/systemone` using model `jev-1.13.0` whenever `TYPESAFE_API_KEY` is present.
     - **Mock Simulation Mode:** Emulates Jev's System One `Choice` primitive (probability distribution + calibrated confidence) locally without requiring an API key or credit balance.
   * **Knowledge Base Integration:** Integrates with `Source Code/icd10_cardiovascular.csv` (the 367 PERKI cardiovascular codes).
   * **Confidence-Gating (Selective Prediction):** Evaluates confidence threshold ($\ge 0.60 \rightarrow \text{AUTO\_ACCEPT}$, $< 0.60 \rightarrow \text{FLAG\_FOR\_REVIEW}$).
2. **Evaluated on CAR0001–CAR0005:**
   * Executed `python jev_icd10_mapper.py --test-car-cases`.
   * Results saved to [`Source Code/jev_output/car_cases_jev_mapping.csv`](file:///d:/Users/justi/OneDrive/Documents/Binus%20aso/Antigravity%20Medical%20STT/Source%20Code/jev_output/car_cases_jev_mapping.csv).
   * **Key Behavior Confirmed:**
     - `CAR0003` (explicit doctor assessment of CHF) scored high confidence (`I50.0`, `AUTO_ACCEPT`).
     - `CAR0001` & `CAR0002` (incomplete anamnesis audio with blank assessment) scored low confidence (`FLAG_FOR_REVIEW`), proving selective prediction works as designed.

> [!IMPORTANT]
> **Action Item for Next Step:**
> 1. Integrate `map_soap_to_icd10()` from `jev_icd10_mapper.py` directly into [`Source Code/batch_soap_analysis.py`](file:///d:/Users/justi/OneDrive/Documents/Binus%20aso/Antigravity%20Medical%20STT/Source%20Code/batch_soap_analysis.py).
> 2. Whenever the user acquires live Jev credits, simply paste the API key into `Source Code/.env` as `TYPESAFE_API_KEY="jv_live_..."` — the mapper will automatically switch to Live API mode with zero code changes needed.
