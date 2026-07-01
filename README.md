# AI-Assisted ICU Case Review System

This Software_System deliverable is a standalone academic information system for a Case Review Coordinator.

It helps users prioritize ICU cases, create review tasks, add new case submissions, run AI risk analysis, inspect explainability/similar cases/clustering/anomaly/uncertainty, update review status and priority, save notes, track audit history, and print case summaries.

## Safety Scope

Academic decision-support prototype only. It does not diagnose, prescribe medication, recommend treatment, or replace clinical staff.

## Local Storage

The system automatically creates local SQLite storage at:

```text
data/software_system.db
```

Tables include review items, manual cases, review notes, and audit events. No external database setup is required. Raw MIMIC files are not included or processed.

## Real AI Features

- Trained supervised ML risk prediction from `models/ai_risk_model.pkl`
- Explainable AI from saved model feature importance
- KNN similar case retrieval
- KMeans clustering
- IsolationForest anomaly detection
- Uncertainty estimation
- Optional live generative AI only when `LLM_API_KEY` is configured
- AI Model Report page using saved metadata and metrics

## Run

Git Bash:

```bash
cd /c/projects/Final_Project_Clean/Software_System
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
./.venv/Scripts/python.exe app.py
```

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
- Read/filter queue items in Review Queue.
- Update status, priority, and assigned reviewer from Review Item pages.
- Add review notes.
- Archive/delete local workflow records. Delete requests archive records by default and never delete dataset rows.
- Review audit events for traceability.


