import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

NUMERIC_COLS = [
    'grade', 'age', 'household_income', 'attendance_rate',
    'academic_score', 'distance_to_school_km', 'school_infra_score'
]
CATEGORICAL_COLS = [
    'district', 'area_type', 'gender', 'socioeconomic_status', 'parent_education', 'midday_meal'
]

def load_data(path: str) -> pd.DataFrame:
    """Load xlsx or csv and return a DataFrame."""
    if path.endswith('.xlsx') or path.endswith('.xls'):
        df = pd.read_excel(path, sheet_name=0)
    else:
        df = pd.read_csv(path)
    return df

def clean_and_engineer(df: pd.DataFrame, fit_encoder=None):
    """Cleans dataset, fills missing values, encodes categories.
       Returns X, y, encoder (encoder only when fitting)."""
    # Standardize column names to lower
    df.columns = [c.strip() for c in df.columns]
    # Required cols check - may adapt
    # Fill numeric missing with median
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    # Binary midday_meal normalization
    if 'midday_meal' in df.columns:
        df['midday_meal'] = df['midday_meal'].apply(lambda x: 1 if str(x) in ['1','True','true','yes','Yes'] else 0)

    # Categorical missing fill
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('Unknown')

    # Label
    y = None
    if 'dropout_next_year' in df.columns:
        y = df['dropout_next_year'].astype(int)

    # One hot encode categorical columns
    encoder = fit_encoder
    cat_df = pd.DataFrame()
    if fit_encoder is None:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(df[[c for c in CATEGORICAL_COLS if c in df.columns]])
    cat_names = encoder.get_feature_names_out([c for c in CATEGORICAL_COLS if c in df.columns])
    cat_vals = encoder.transform(df[[c for c in CATEGORICAL_COLS if c in df.columns]])
    cat_df = pd.DataFrame(cat_vals, columns=cat_names, index=df.index)

    # Prepare final X
    nums = df[[c for c in NUMERIC_COLS if c in df.columns]]
    X = pd.concat([nums, cat_df], axis=1)
    return X, y, encoder