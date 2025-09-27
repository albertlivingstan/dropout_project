AI-based Student Dropout Prediction & Counseling Platform

An end-to-end solution built for Smart India Hackathon (SIH) 2025 to help the Government of Rajasthan predict and prevent student dropouts in government schools.
The system uses Machine Learning and Multilingual AI Counseling to support teachers, administrators, and parents.

🚀 Features
Dropout Risk Prediction from existing student data (attendance, scores, income, etc.)
Explainable AI – highlights top features influencing predictions
Multilingual Counseling Messages in English, Hindi, and Rajasthani
Role-Based Dashboard for Teachers & Parents
FastAPI Backend + Streamlit Frontend for a smooth demo
CSV Upload & Download of student predictions
📂 Project Structure
dropout_project/
│
├── backend/           # FastAPI backend + model training
│   ├── train.py
│   ├── data_processing.py
│   ├── utils.py
│   ├── app_api.py
│
├── frontend/          # Streamlit dashboard
│   └── streamlit_app.py
│
├── data/              # Input CSV/Excel files
├── models/            # Trained model + encoders
├── notebooks/         # Jupyter/Colab exploration
├── venv/              # Virtual environment (local only)
└── README.md

🛠️ Installation & Setup
1.⁠ ⁠Clone Repo
git clone: https://github.com/albertlivingstan/dropout_project.git
cd dropout_project

2.⁠ ⁠Create Virtual Environment
python -m venv venv
# Activate venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# Linux/Mac
source venv/bin/activate

3.⁠ ⁠Install Dependencies
pip install -r requirements.txt


(Or directly install: pandas, numpy, scikit-learn, xgboost, fastapi, uvicorn, joblib, deep-translator, streamlit, plotly, requests, openpyxl)

📊 Training the Model

Place your dataset in data/ (Excel or CSV with required columns).

Run:

cd backend
python train.py --input ../data/rajasthan_students.xlsx



This generates:

models/trained_model.pkl
models/encoder.pkl
⚡ Running the Backend (API)
cd backend
uvicorn app_api:[app --host 0.0.0.0 --port 8000 --reload](http://127.0.0.1:8000/docs#/)

API Docs: http://127.0.0.1:8000/docs
Test /health endpoint → should return { "status": "ok" }
📈 Running the Frontend (Dashboard)

Open a new terminal:

cd frontend
streamlit run streamlit_app.py


Go to: [http://localhost:8501](http://localhost:8501/)

Demo Flow:

Upload sample_students.csv
Click Run Predictions
View Top At-Risk Students + counseling in 3 languages
Download results as CSV
🧪 Sample Data

We’ve included a sample file: sample_students.csv

10 students with mixed dropout risk levels
Useful for quick testing in frontend
📊 Example Prediction (API Response)
{
  "student_id": "RJ000001",
  "risk_score": 0.92,
  "advice": {
    "en": "High risk of dropout. Immediate intervention recommended...",
    "hi": "ड्रॉपआउट का उच्च जोखिम...",
    "rj": "ड्रॉपआउट का उच्च जोखिम... (Rajasthani phrasing suggested)"
  }
}

📚 References
UDISE+ 2022-23 – National education data
ShalaDarpan Rajasthan – State MIS
ASER 2023 Report – Learning outcomes
ScienceDirect / IEEE papers on dropout prediction
AI4Bharat IndicTrans2 – Multilingual NLP
👥 Team
[Your Team Name]
Built for Smart India Hackathon 2025 (Govt. of Rajasthan Problem Statement)
Contributors: [List team members]
