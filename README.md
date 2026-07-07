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

## Real AI Components

- Supervised ML risk prediction from `models/ai_risk_model.pkl`
- Explainable AI using saved model feature importance
- KNN similar-case retrieval
- KMeans clustering
- IsolationForest anomaly detection
- Uncertainty estimation using probability, entropy, threshold distance, and data quality
- Optional live generative AI with `LLM_API_KEY`
- AI Model Report using saved metadata, transparency information, and real metrics

## Run

PowerShell:

```powershell
cd C:\projects\Final_Project_Clean\Software_System
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
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
