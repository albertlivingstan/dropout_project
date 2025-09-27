import streamlit as st
import pandas as pd
import requests
import os
import joblib
import json
from pathlib import Path

# Config
API_URL = "http://127.0.0.1:8000/predict"
# MODEL_LOCAL = "../models/trained_model.pkl"
# ENCODER_LOCAL = "../models/encoder.pkl"


if st.button("Run Predictions via API"):
    results = []
    for _, row in df.iterrows():
        student_data = row.to_dict()
        response = requests.post(API_URL, json=student_data)
        if response.status_code == 200:
            results.append(response.json())
    results_df = pd.DataFrame(results)
    st.write(results_df.head())

st.set_page_config(page_title="Rajasthan Dropout Dashboard", layout="wide")

st.title("AI Dropout Prediction & Counseling — Rajasthan")

uploaded_file = st.file_uploader("Upload student Excel/CSV (with labels or without)", type=['xlsx','csv'])

use_local_predict = st.checkbox("Use local model (no API)", value=True)

if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    st.subheader("Preview data")
    st.dataframe(df.head())

    if st.button("Run predictions for all rows"):
        # For speed, if local model preferred, load and do batch
        if use_local_predict:
            st.info("Using local model for batch prediction")
            # use backend utils to preprocess + predict
            import sys
            sys.path.append('../backend')
            from utils import load_model
            from data_processing import clean_and_engineer
            model, encoder = load_model()
            X, _, _ = clean_and_engineer(df, fit_encoder=encoder)
            df['dropout_risk'] = model.predict_proba(X)[:,1]
            # Generate messages
            df['counseling_en'] = df['dropout_risk'].apply(lambda r: ("High risk..." if r>=0.7 else ("Moderate risk.." if r>=0.4 else "Low risk...")) )
        else:
            st.info("Using API for per-row predictions (slower)")
            results = []
            for _, row in df.iterrows():
                payload = row.to_dict()
                try:
                    r = requests.post(API_URL, json=payload, timeout=10)
                    res = r.json()
                    results.append(res)
                except Exception as e:
                    st.error(f"API error: {e}")
                    results.append({"risk_score": None})
            df['dropout_risk'] = [r.get('risk_score') for r in results]
            df['counseling_en'] = [r.get('advice',{}).get('en') for r in results]

        st.success("Predictions finished.")
        st.subheader("Top 10 At-Risk Students")
        top10 = df.sort_values('dropout_risk', ascending=False).head(10)
        st.table(top10[['student_id','district','grade','attendance_rate','academic_score','dropout_risk']])

        st.markdown("## Counseling Samples (English)")
        for idx, row in top10.iterrows():
            st.write(f"*{row.get('student_id','-')}* — Risk: {row['dropout_risk']:.2f}")
            st.write(row.get('counseling_en','No message'))

        # allow downloading predictions
        out_path = "predictions_with_risks.csv"
        df.to_csv(out_path, index=False)
        st.download_button("Download predictions CSV", data=open(out_path,'rb'), file_name="predictions_with_risks.csv")

else:
    st.info("Upload your Rajasthan student file to start predictions.")