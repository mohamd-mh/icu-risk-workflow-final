# AI-Assisted ICU Quality Review System

This Software_System deliverable is a standalone academic information system for an ICU Quality & Patient Safety Coordinator.

## Target User

The target user is an ICU Quality & Patient Safety Coordinator. This is a real-life inspired role based on hospital quality and patient-safety, mortality review, clinical quality, clinical data analysis, and utilization review workflows.

The system helps the user prioritize ICU case review, document notes, track status, and maintain auditability using real AI/ML outputs. The user supports case tracking, audit preparation, and follow-up coordination and does not make treatment decisions.

## Safety Scope

Academic quality-review and documentation prototype only. It does not diagnose, prescribe medication, recommend treatment, or replace qualified clinical staff.

## Local Storage

The system automatically creates local SQLite storage at `data/software_system.db`. Tables include review items, manual cases, review notes, and audit events. No external database setup is required. Raw MIMIC files are not included or processed.

## Data and Feature Basis

- The software uses the processed academic ICU dataset `data/patient_features_ai.csv`; no raw MIMIC files are included.
- Selected early ICU features cover demographic and admission context, physiology, and laboratory values.
- The training target is `hospital_expire_flag`, used only as the label and never as a prediction input.
- Leakage and identifier columns are excluded: `subject_id`, `hadm_id`, `icustay_id`, and `hospital_expire_flag`.
- Features were chosen because these feature families are common in ICU severity and prediction settings. Individual values do not manually determine risk; the trained ML model learns statistical patterns from processed data.
- If processed MIMIC-derived data is shared, keep the repository private or limited to the course in accordance with the applicable data-use terms.

## Dataset Provenance

- `data/patient_features_ai.csv` is a processed, ML-ready feature table derived from the **full MIMIC-III v1.4 dataset** provided by the course via a shared Google Drive folder (not a separate, smaller, or unofficial data source).
- Source tables used to build the feature table: `PATIENTS.csv`, `ADMISSIONS.csv`, `ICUSTAYS.csv`, `CHARTEVENTS.csv`, `LABEVENTS.csv`, `D_ITEMS.csv`, and `D_LABITEMS.csv`.
- Raw MIMIC-III files are **not included** in this repository/submission package, primarily because of size - `CHARTEVENTS.csv` alone is approximately 33GB. The processed feature table (`data/patient_features_ai.csv`) is included in its place and is what the software actually reads and trains on.
- The submission package includes the processed feature table used by the prototype. The original extraction script and raw MIMIC tables are not included.
- Target label: `hospital_expire_flag` (used only for training/evaluation, never as a model input).
- Excluded leakage/identifier columns (never used as model inputs): `subject_id`, `hadm_id`, `icustay_id`, `hospital_expire_flag`.
- Missing values are not hand-filled; they are handled by the preprocessing/model pipeline (median imputation for numeric features, most-frequent imputation for categorical features, as implemented in `ml/train_ai_model.py`).
- Because this processed table is derived from restricted-access MIMIC-III data, it should remain course-limited/private and not be redistributed outside the academic submission in accordance with the applicable data-use terms.

## Data Preprocessing Pipeline

The processed feature table (`patient_features_ai.csv`, 1000 ICU stays) was derived from the full MIMIC-III v1.4 dataset provided via the course's shared Google Drive folder. The extraction process read `PATIENTS.csv`, `ADMISSIONS.csv`, and `ICUSTAYS.csv` for cohort, demographics, and the `hospital_expire_flag` label; `CHARTEVENTS.csv` (~33GB, processed in chunks due to its size) for vital signs (heart rate, systolic BP, respiratory rate, temperature, SpO2); and `LABEVENTS.csv` (processed in chunks) for lab values (creatinine, glucose, WBC, hemoglobin). The original extraction script is not included in this submission package; the processed feature table is included instead.

## Baseline Methodology

Baseline uses the SIRS (Systemic Inflammatory Response Syndrome) criteria (Bone et al., 1992). See `baseline/risk_rules.py` for the 4 cited criteria and thresholds:

1. Temperature > 38C or < 36C
2. Heart rate > 90/min
3. Respiratory rate > 20/min
4. White blood cell count > 12,000/mm3 or < 4,000/mm3

SIRS-positive is the standard clinical definition of 2 or more criteria met. This project maps criteria count to the existing Low/Medium/High scheme as: 0-1 criteria -> Low, 2 criteria -> Medium (SIRS-positive), 3-4 criteria -> High.

Citation: Bone, R.C., Balk, R.A., Cerra, F.B., et al. (1992). "Definitions for sepsis and organ failure and guidelines for the use of innovative therapies in sepsis." *Chest*, 101(6), 1644-1655.

The other clinical variables in this dataset (systolic blood pressure, creatinine, glucose, hemoglobin, SpO2, age, admission type) are **not** part of the official SIRS definition and are intentionally excluded from the baseline. They remain used by the trained ML model (all 13 input features) and by the separate "weighted risk assessor" stage inside the modular workflow (`agents/multi_agent_workflow.py`). That stage extends beyond SIRS with additional physiological and laboratory variables inspired by general ICU severity-scoring concepts (e.g., APACHE II, SOFA); its specific weights (1-3 per factor) reflect engineering judgment and are not calibrated against this dataset or drawn from a specific published scoring table. This is a stated limitation, not a validated clinical score.

## Real AI Components

- Supervised ML risk prediction from `models/ai_risk_model.pkl`
- Explainable AI using saved model feature importance
- KNN similar-case retrieval
- KMeans clustering
- IsolationForest anomaly detection
- Uncertainty estimation using probability, entropy, threshold distance, and data quality
- Optional live generative AI with `LLM_API_KEY`
- AI Model Report using saved metadata, transparency information, and real metrics

## Workflow Design Note

The implemented prototype is not a full autonomous AutoGen-style multi-agent
negotiation system. It uses a modular sequence of specialized components for
prediction, verification, explainability, similarity, anomaly detection,
uncertainty estimation, and review documentation. The "baseline" and
"multi-agent workflow" pipelines are both rule-based scoring components; the
only real LLM call in the system is the optional generative-AI case-summary
assistant, which is used only when `LLM_API_KEY` is configured and otherwise
falls back to an offline templated summary built from local model outputs.

## Run

Clone or open this repository, then create a virtual environment named `.venv` in the project root.

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/`.

## Main Routes

```text
/
/software
/icu-dashboard
/icu-cases
/icu-cases/200024
/review-queue
/new-case
/ai-evaluation
/user-guide
```

## CRUD Workflow

- Create review items from dataset cases using Add to Review Queue.
- Create manual case submissions from Add New Case.
- Read and filter queue items in the Quality Review Queue.
- Update status, review priority, and assigned coordinator from Review Item pages.
- Add review notes and follow-up documentation.
- Archive/delete local workflow records. Delete requests archive records by default and never delete dataset rows.
- Review audit events for traceability and audit preparation.
