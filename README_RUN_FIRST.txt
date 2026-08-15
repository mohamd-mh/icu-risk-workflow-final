AI-Assisted ICU Quality Review Workbench
Run this first after downloading Software_System.zip

Windows
1. Extract Software_System.zip
2. Open terminal in Software_System
3. Run:
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python app.py
4. Open:
   http://127.0.0.1:5000

Mac/Linux
1. Extract Software_System.zip
2. Open terminal in Software_System
3. Run:
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python app.py
4. Open:
   http://127.0.0.1:5000

Launcher scripts
- Windows users can also run run_windows.bat from the Software_System folder.
- Mac/Linux users can also run:
  bash run_mac_linux.sh

Important notes
- The app uses the included processed file data/patient_features_ai.csv.
- Raw MIMIC files are not required for running the demo.
- The local SQLite review database is created automatically in the data folder.
- The app is an academic prototype for quality-review workflow only.
