from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from utils import predict_risk, counseling_message_en, translate_text

app = FastAPI(title=" Dropout Prediction API")

class StudentData(BaseModel):
    student_id: Optional[str]
    district: Optional[str]
    area_type: Optional[str]
    grade: Optional[int]
    age: Optional[int]
    gender: Optional[str]
    socioeconomic_status: Optional[str]
    parent_education: Optional[str]
    household_income: Optional[float]
    attendance_rate: Optional[float]
    academic_score: Optional[float]
    distance_to_school_km: Optional[float]
    midday_meal: Optional[int]
    school_infra_score: Optional[float]

@app.post("/predict")
def predict(student: StudentData):
    rec = student.dict()
    try:
        risk = predict_risk(rec)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    reasons = []  # optional: compute top contributing features using SHAP or feature importance
    advice_en = counseling_message_en(risk, top_reasons=reasons)
    advice_hi = translate_text(advice_en, 'hi')
    advice_rj = advice_hi + " (Rajasthani phrasing suggested)"  # placeholder

    return {
        "student_id": rec.get("student_id"),
        "risk_score": risk,
        "advice": {
            "en": advice_en,
            "hi": advice_hi,
            "rj": advice_rj
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}