// ── Auth: redirect to login on 401 ──────────────────────────────────────────
const _originalFetch = window.fetch.bind(window);
window.fetch = async function(...args) {
    const response = await _originalFetch(...args);
    if (response.status === 401) {
        window.location.href = '/login';
        return response;
    }
    return response;
};

// ── Logout button ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const btnLogout = document.getElementById('btn-logout');
    const themeToggleButton = document.getElementById('theme-toggle');

    if (btnLogout) {
        btnLogout.addEventListener('click', async () => {
            await fetch('/api/logout', { method: 'POST', credentials: 'include' });
            window.location.href = '/login';
        });
    }

    if (themeToggleButton) {
        themeToggleButton.addEventListener('click', toggleTheme);
    }

    loadThemePreference();
});

// Check for browser support
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

let recognition = null;
if (!SpeechRecognition) {
    console.warn("Web Speech API not supported. Recording disabled.");
} else {
    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
}

// DOM Elements
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const langSelect = document.getElementById('lang-select');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const finalTextEl = document.getElementById('final-text');
const interimTextEl = document.getElementById('interim-text');
const transcriptBox = document.getElementById('transcript-box');
const loadingOverlay = document.getElementById('loading-overlay');
const reportContent = document.getElementById('report-content');
const btnDownload = document.getElementById('btn-download');
const btnEdit = document.getElementById('btn-edit');
const reportActions = document.getElementById('report-actions');
const btnMapIcd10 = document.getElementById('btn-map-icd10');
const icd10Loading = document.getElementById('icd10-loading');
const icd10Content = document.getElementById('icd10-content');

const THEME_STORAGE_KEY = 'medical-stt-theme';
const DEFAULT_THEME = 'dark';

function updateThemeToggleLabel(theme) {
    const themeToggle = document.getElementById('theme-toggle');
    if (!themeToggle) return;
    themeToggle.innerHTML = theme === 'light'
        ? '<i class="fa-solid fa-moon"></i> Dark Mode'
        : '<i class="fa-solid fa-sun"></i> Light Mode';
}

function applyTheme(theme) {
    document.body.setAttribute('data-theme', theme);
    updateThemeToggleLabel(theme);
}

function loadThemePreference() {
    const savedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
    const theme = savedTheme === 'light' ? 'light' : DEFAULT_THEME;
    applyTheme(theme);
}

function saveThemePreference(theme) {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
}

function toggleTheme() {
    const current = document.body.getAttribute('data-theme') === 'light' ? 'light' : DEFAULT_THEME;
    const nextTheme = current === 'light' ? 'dark' : 'light';
    applyTheme(nextTheme);
    saveThemePreference(nextTheme);
}

let finalTranscript = '';
let currentSoapData = null;
let isRecording = false;
let isEditMode = false;

// Speech Recognition Events
if (recognition) {
    recognition.onstart = function() {
        isRecording = true;
        statusDot.classList.add('recording');
        statusText.textContent = 'Recording...';
        btnStart.disabled = true;
        btnStop.disabled = false;
        
        // Clear previous if any
        finalTranscript = '';
        finalTextEl.innerHTML = '';
        interimTextEl.innerHTML = '';
        
        // Remove placeholder
        const placeholder = transcriptBox.querySelector('.placeholder');
        if (placeholder) placeholder.style.display = 'none';
    };

    recognition.onresult = function(event) {
        let interimTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript + ' ';
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }
        
        finalTextEl.innerHTML = finalTranscript;
        interimTextEl.innerHTML = interimTranscript;
        
        // Auto-scroll to bottom
        transcriptBox.parentElement.scrollTop = transcriptBox.parentElement.scrollHeight;
    };

    recognition.onerror = function(event) {
        console.error("Speech recognition error", event.error);
        statusText.textContent = 'Error: ' + event.error;
        stopRecording();
    };

    recognition.onend = function() {
        if (isRecording) {
            recognition.start();
        }
    };
}

function stopRecording() {
    isRecording = false;
    if (recognition) recognition.stop();
    statusDot.classList.remove('recording');
    statusText.textContent = 'Analysis mode';
    btnStart.disabled = false;
    btnStop.disabled = true;
}

// Button Listeners
btnStart.addEventListener('click', () => {
    if (!recognition) {
        alert('Speech recognition is not supported in this browser. Please use Chrome or Edge.');
        return;
    }
    recognition.lang = langSelect.value;
    // Reset ICD-10 panel for new recording
    if (btnMapIcd10) {
        btnMapIcd10.disabled = true;
        btnMapIcd10.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Map to ICD-10';
        btnMapIcd10.title = "Generate SOAP report first";
    }
    if (icd10Content) {
        icd10Content.innerHTML = '<p class="placeholder icd10-placeholder"><i class="fa-solid fa-stethoscope"></i> Complete consultation and analyze SOAP report, then click "Map to ICD-10".</p>';
    }
    try {
        recognition.start();
    } catch (e) {
        console.error(e);
    }
});

btnStop.addEventListener('click', () => {
    stopRecording();
    
    // Flush any remaining interim text to final
    if (interimTextEl.innerHTML.trim() !== '') {
        finalTranscript += interimTextEl.innerHTML + ' ';
        finalTextEl.innerHTML = finalTranscript;
        interimTextEl.innerHTML = '';
    }
    
    const textToAnalyze = finalTranscript.trim();
    if (textToAnalyze) {
        analyzeText(textToAnalyze);
    } else {
        alert("No text was recorded.");
        statusText.textContent = 'Ready';
    }
});

btnDownload.addEventListener('click', () => {
    if (!currentSoapData) return;
    // If in edit mode, sync edits to currentSoapData first
    if (isEditMode) {
        syncEditsToData();
    }
    downloadCsv(currentSoapData);
});

// Edit / Done Editing toggle
btnEdit.addEventListener('click', () => {
    if (!isEditMode) {
        enterEditMode();
    } else {
        exitEditMode();
    }
});

// ── Edit Mode Logic ──────────────────────────────────────────────────────────

function enterEditMode() {
    isEditMode = true;
    btnEdit.innerHTML = '<i class="fa-solid fa-check"></i> Done Editing';
    btnEdit.classList.remove('secondary');
    btnEdit.classList.add('edit-active');

    // Make all soap-value elements contenteditable
    reportContent.querySelectorAll('.soap-value').forEach(el => {
        el.contentEditable = 'true';
        el.classList.add('editable');
        el.setAttribute('spellcheck', 'false');
    });

    // Show the edit hint banner
    let hint = reportContent.querySelector('.edit-hint');
    if (!hint) {
        hint = document.createElement('div');
        hint.className = 'edit-hint';
        hint.innerHTML = '<i class="fa-solid fa-circle-info"></i> Click any field to edit. Changes will be included in the CSV export.';
        reportContent.prepend(hint);
    }
    hint.style.display = 'flex';
}

function exitEditMode() {
    isEditMode = false;
    btnEdit.innerHTML = '<i class="fa-solid fa-pen-to-square"></i> Edit Report';
    btnEdit.classList.add('secondary');
    btnEdit.classList.remove('edit-active');

    // Remove contenteditable
    reportContent.querySelectorAll('.soap-value').forEach(el => {
        el.contentEditable = 'false';
        el.classList.remove('editable');
    });

    // Sync edits back into the data object
    syncEditsToData();

    // Hide the hint
    const hint = reportContent.querySelector('.edit-hint');
    if (hint) hint.style.display = 'none';
}

/**
 * Reads the current text content of each soap-value element in the DOM and
 * writes it back to currentSoapData using the data-field-path attribute.
 */
function syncEditsToData() {
    if (!currentSoapData) return;

    reportContent.querySelectorAll('.soap-value[data-field-path]').forEach(el => {
        const path = el.getAttribute('data-field-path');
        const newValue = el.innerText.trim();
        setNestedValue(currentSoapData, path, newValue);
    });
}

/**
 * Sets a value in a nested object using a dot-separated path string.
 * e.g. setNestedValue(obj, "subjective.chief_complaint", "headache")
 */
function setNestedValue(obj, path, value) {
    const keys = path.split('.');
    let current = obj;
    for (let i = 0; i < keys.length - 1; i++) {
        if (current[keys[i]] === undefined || current[keys[i]] === null) return;
        current = current[keys[i]];
    }
    const lastKey = keys[keys.length - 1];
    // Preserve array type if the original was an array
    if (Array.isArray(current[lastKey])) {
        // Split by comma if the user typed a comma-separated list
        current[lastKey] = value.split(',').map(s => s.trim()).filter(s => s !== '');
    } else {
        current[lastKey] = value;
    }
}

// ── API Calls ────────────────────────────────────────────────────────────────

async function analyzeText(text) {
    loadingOverlay.style.display = 'flex';
    reportActions.style.display = 'none';

    // Exit edit mode if we're re-analyzing
    if (isEditMode) {
        isEditMode = false;
        btnEdit.innerHTML = '<i class="fa-solid fa-pen-to-square"></i> Edit Report';
        btnEdit.classList.add('secondary');
        btnEdit.classList.remove('edit-active');
    }
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }
        
        const data = await response.json();
        currentSoapData = data;
        renderSoapReport(data);
        reportActions.style.display = 'flex';
        statusText.textContent = 'Analysis Complete';

        // Enable ICD-10 mapping button
        if (btnMapIcd10) {
            btnMapIcd10.disabled = false;
            btnMapIcd10.title = "Click to map SOAP report to ICD-10 using Jev System One";
        }
        if (icd10Content && !icd10Content.querySelector('.icd10-result-card')) {
            icd10Content.innerHTML = '<p class="placeholder icd10-placeholder" style="color: var(--primary);"><i class="fa-solid fa-wand-magic-sparkles"></i> SOAP report ready! Click "Map to ICD-10" above to predict diagnostic code.</p>';
        }
    } catch (error) {
        console.error("Analysis failed", error);
        reportContent.innerHTML = `<p style="color: var(--danger)">Error during analysis: ${error.message}</p>`;
        statusText.textContent = 'Error';
    } finally {
        loadingOverlay.style.display = 'none';
    }
}

async function downloadCsv(soapData) {
    try {
        const response = await fetch('/api/download_csv', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ soap_data: soapData })
        });
        
        if (!response.ok) {
            throw new Error(`Server error generating CSV`);
        }
        
        const csvText = await response.text();
        
        // Create a blob and trigger download
        const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement("a");
        const url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        
        const date = new Date();
        const ts = `${date.getDate()}_${date.getMonth()+1}_${date.getFullYear()}_${date.getHours()}_${date.getMinutes()}`;
        link.setAttribute("download", `medical_report_live_${ts}.csv`);
        
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
    } catch (error) {
        console.error("Download failed", error);
        alert("Failed to generate CSV.");
    }
}

// ── Render Logic ─────────────────────────────────────────────────────────────

function renderSoapReport(data) {
    let html = `
        <div class="soap-section">
            <div class="soap-header">Overview</div>
            <div class="soap-content">
                ${makeRow('Summary', data.summary, 'summary')}
                ${makeRow('Encounter Type', data.encounter_type, 'encounter_type')}
                ${makeRow('Confidence', data.extraction_confidence, 'extraction_confidence')}
            </div>
        </div>
        
        <div class="soap-section">
            <div class="soap-header">Subjective</div>
            <div class="soap-content">
                ${makeRow('Chief Complaint', data.subjective?.chief_complaint, 'subjective.chief_complaint')}
                ${makeRow('Symptoms', data.subjective?.symptoms, 'subjective.symptoms')}
                ${makeRow('Onset', data.subjective?.symptom_onset, 'subjective.symptom_onset')}
                ${makeRow('Medical History', data.subjective?.medical_history, 'subjective.medical_history')}
                ${makeRow('Medications', data.subjective?.current_medications, 'subjective.current_medications')}
                ${makeRow('Allergies', data.subjective?.allergies, 'subjective.allergies')}
            </div>
        </div>
        
        <div class="soap-section">
            <div class="soap-header">Objective</div>
            <div class="soap-content">
                ${makeRow('Vitals', data.objective?.vital_signs, 'objective.vital_signs')}
                ${makeRow('Physical Exam', data.objective?.physical_exam_findings, 'objective.physical_exam_findings')}
            </div>
        </div>
        
        <div class="soap-section">
            <div class="soap-header">Assessment & Plan</div>
            <div class="soap-content">
                ${makeRow('Diagnosis', data.assessment?.diagnosis, 'assessment.diagnosis')}
                ${makeRow('Differentials', data.assessment?.differential_diagnosis, 'assessment.differential_diagnosis')}
                ${makeRow('Prescriptions', data.plan?.prescribed_medications, 'plan.prescribed_medications')}
                ${makeRow('Treatment Plan', data.plan?.treatment_plan, 'plan.treatment_plan')}
                ${makeRow('Investigations', data.plan?.investigations_ordered, 'plan.investigations_ordered')}
                ${makeRow('Referrals', data.plan?.referrals, 'plan.referrals')}
                ${makeRow('Follow up', data.plan?.follow_up, 'plan.follow_up')}
            </div>
        </div>
    `;
    
    reportContent.innerHTML = html;
}

/**
 * Renders a single label/value row.
 * @param {string} label   - Display label
 * @param {*}      value   - The data value (string, array, or null)
 * @param {string} path    - Dot-separated path used to write back edits (e.g. "subjective.chief_complaint")
 */
function makeRow(label, value, path) {
    if (value === null || value === undefined || value === '') return '';
    if (Array.isArray(value)) {
        if (value.length === 0) return '';
        value = value.join(', ');
    }
    return `
        <div class="soap-row">
            <div class="soap-label">${label}</div>
            <div class="soap-value" data-field-path="${path}">${value}</div>
        </div>
    `;
}

// ── ICD-10 Mapping Logic ─────────────────────────────────────────────────────

if (btnMapIcd10) {
    btnMapIcd10.addEventListener('click', async () => {
        if (!currentSoapData) {
            alert('Please generate a SOAP report first.');
            return;
        }
        if (isEditMode) {
            syncEditsToData();
        }
        await mapSoapToIcd10(currentSoapData);
    });
}

async function mapSoapToIcd10(soapData) {
    if (!icd10Loading || !icd10Content) return;
    icd10Loading.style.display = 'flex';
    icd10Content.style.display = 'none';
    if (btnMapIcd10) btnMapIcd10.disabled = true;

    try {
        const response = await fetch('/api/map-icd10', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ soap_data: soapData })
        });

        if (!response.ok) {
            const errJson = await response.json().catch(() => ({}));
            throw new Error(errJson.error || `Server error: ${response.statusText}`);
        }

        const data = await response.json();
        renderIcd10Result(data);
        if (btnMapIcd10) {
            btnMapIcd10.innerHTML = '<i class="fa-solid fa-rotate-right"></i> Re-Map';
        }
    } catch (err) {
        console.error("ICD-10 mapping error", err);
        icd10Content.innerHTML = `<p style="color: var(--danger); text-align: center; font-size: 0.85rem;"><i class="fa-solid fa-triangle-exclamation"></i> ${escapeHtml(err.message)}</p>`;
        icd10Content.style.display = 'block';
        if (btnMapIcd10) {
            btnMapIcd10.innerHTML = '<i class="fa-solid fa-rotate-right"></i> Retry';
        }
    } finally {
        icd10Loading.style.display = 'none';
        if (btnMapIcd10) btnMapIcd10.disabled = false;
    }
}

function renderIcd10Result(data) {
    if (!icd10Content) return;
    const isAccepted = data.status === 'AUTO_ACCEPT';
    const confVal = typeof data.confidence === 'number' ? data.confidence : 0.0;
    const confPct = Math.round(confVal * 100);
    const confDisplay = (confVal * 100).toFixed(1);

    const html = `
        <div class="icd10-result-card">
            <div class="icd10-header-row">
                <span class="icd10-code-badge">${escapeHtml(data.code || 'N/A')}</span>
                <span class="icd10-decision-pill ${isAccepted ? 'decision-accept' : 'decision-review'}">
                    <i class="fa-solid ${isAccepted ? 'fa-circle-check' : 'fa-triangle-exclamation'}"></i>
                    ${isAccepted ? 'Auto-Accepted' : 'Flagged for Review'}
                </span>
            </div>
            <div class="icd10-desc-title">${escapeHtml(data.description || 'Unspecified Condition')}</div>
            <div class="icd10-confidence-wrap">
                <div class="confidence-labels">
                    <span>Model Confidence (Calibrated)</span>
                    <span class="confidence-val">${confDisplay}%</span>
                </div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill ${isAccepted ? '' : 'bar-warning'}" style="width: ${Math.max(6, confPct)}%"></div>
                </div>
            </div>
            <div class="icd10-footer-meta">
                <span><i class="fa-solid fa-brain"></i> ${escapeHtml(data.mode || 'Jev System One')}</span>
                ${data.detected_language === 'id' ? '<span title="Indonesian clinical concepts normalized to canonical English ICD-10 ontology"><i class="fa-solid fa-language"></i> ID → EN Normalized</span>' : '<span><i class="fa-solid fa-globe"></i> English</span>'}
                <span>Threshold: ≥60%</span>
            </div>
        </div>
    `;
    icd10Content.innerHTML = html;
    icd10Content.style.display = 'block';
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

