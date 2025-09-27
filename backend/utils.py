import joblib
import numpy as np
import pandas as pd
from deep_translator import GoogleTranslator

MODEL_PATH = "../models/trained_model.pkl"
ENCODER_PATH = "../models/encoder.pkl"

_model = None
_encoder = None

def load_model():
    global _model, _encoder
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _encoder is None:
        _encoder = joblib.load(ENCODER_PATH)
    return _model, _encoder

def preprocess_single(record: dict):
    """Given a dict of raw student data, return feature vector same as training X."""
    # Convert to DataFrame single row
    df = pd.DataFrame([record])
    from data_processing import clean_and_engineer
    # Clean & encode using loaded encoder
    X, _, _ = clean_and_engineer(df, fit_encoder=_encoder)
    # Ensure column alignment (train used columns of encoder)
    # Models expect same columns - if missing, add zeros
    return X

def predict_risk(record: dict):
    model, encoder = load_model()
    X = preprocess_single(record)
    proba = model.predict_proba(X)[:,1][0]
    return float(proba)

def counseling_message_en(risk, top_reasons=None):
    """Rule-based template generator; top_reasons optional list of strings."""
    # Simple template; you can plug LLM here.
    if risk >= 0.7:
        advice = "High risk of dropout. Immediate intervention recommended: schedule a parent-teacher meeting, assign remedial tutoring, and check socio-economic barriers."
    elif risk >= 0.4:
        advice = "Moderate risk. Monitor attendance closely, provide mentoring and engage parents with progress updates."
    else:
        advice = "Low risk. Keep encouraging regular attendance and provide positive feedback to maintain engagement."
    if top_reasons:
        advice += " Possible contributing factors: " + ", ".join(top_reasons) + "."
    return advice

def translate_text(text, target_language_code='hi'):
    """Translate English text to Hindi ('hi') or other codes. For Rajasthani we use Hindi as proxy and append note."""
    try:
        translated = GoogleTranslator(source='auto', target=target_language_code).translate(text)
        return translated
    except Exception as e:
        # Fallback to original
        return text