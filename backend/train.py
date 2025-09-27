import os
import joblib
import argparse
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

from data_processing import load_data, clean_and_engineer

MODEL_OUT = "../models/trained_model.pkl"
ENCODER_OUT = "../models/encoder.pkl"

def train(args):
    df = load_data(args.input)
    X, y, encoder = clean_and_engineer(df, fit_encoder=None)
    if y is None:
        raise ValueError("Input dataset must contain 'dropout_next_year' column for supervised training.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, n_estimators=200)
    model.fit(X_train, y_train)

    # Save model + encoder
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    joblib.dump(encoder, ENCODER_OUT)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to training excel/csv")
    args = parser.parse_args()
    train(args)