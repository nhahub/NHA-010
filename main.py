# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
import sys
import os

# =========================
# 0) Import your utils
# =========================

# adjust path so Python can see src/
sys.path.append("..")

from src.utils.preprocessing import *  # make sure this exists
from src.utils.features_extension import *   # make sure this exists


# =========================
# 1) Load the trained model
# =========================

# Load model relative to main.py
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "notebooks",      # 👈 NOT "..", just "notebooks"
    "churn_model.pkl"
)

MODEL_PATH = os.path.abspath(MODEL_PATH)

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded model from: {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ Error loading churn_model.pkl: {e}")


# =========================
# 2) Input schema (RAW data)
#    Same columns as the original Telco dataset
#    except: we exclude customerID and Churn
# =========================

class CustomerFeatures(BaseModel):
    customerID: object
    gender: object
    SeniorCitizen: int
    Partner: object
    Dependents: object
    tenure: int
    PhoneService: object
    MultipleLines: object
    InternetService: object
    OnlineSecurity: object
    OnlineBackup: object
    DeviceProtection: object
    TechSupport: object
    StreamingTV: object
    StreamingMovies: object
    Contract: object
    PaperlessBilling: object
    PaymentMethod: object
    MonthlyCharges: float
    TotalCharges: float


# =========================
# 3) Expected final feature columns (after preprocessing)
#    Must match the columns used when training the model.
# =========================

EXPECTED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "PaperlessBilling",
    "MonthlyCharges",
    "TotalCharges",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No internet service",
    "OnlineBackup_Yes",
    "DeviceProtection_No internet service",
    "DeviceProtection_Yes",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No internet service",
    "StreamingTV_Yes",
    "StreamingMovies_No internet service",
    "StreamingMovies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "NEW_noProt",
    "NEW_Engaged",
    "NEW_Young_Not_Engaged",
    "NEW_FLAG_ANY_STREAMING",
    "NEW_FLAG_AutoPayment",
    "NEW_TotalServices",
    "NEW_AVG_Service_Fee",
]


# =========================
# 4) Helper: ensure all expected columns exist
# =========================

def ensure_expected_columns(df: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
    """
    Add any missing columns (set to 0) and reorder columns
    to match the training-time feature order.
    """
    df_final = df.copy()

    for col in expected_cols:
        if col not in df_final.columns:
            df_final[col] = 0

    # Keep only expected columns, in exact order
    df_final = df_final[expected_cols]
    return df_final


# =========================
# 5) Preprocessing pipeline for API:
#    RAW → preprocessing utils → feature engineering → align columns
# =========================

def preprocess_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same preprocessing and feature engineering that were used
    during model training, then align to EXPECTED_COLUMNS.
    """
    df = raw_df.copy()
    print("After preprocessing:", df['customerID'].values)
    
    # Your own pipeline (must match notebook training):
    # - data_preprocessing: clean + encode + outliers, etc.
    # - feature_engineering: NEW_... features
    # 1) Fix TotalCharges
    df = convert_to_numeric(df, "TotalCharges", fill_method='median')


    # 2) Drop ID column if exists
    df = drop_column(df, "customerID")
    

    
    

    # 3) Encode categorical features
    # Make sure your encode_categorical_features returns (df_encoded, label_encoders)
    df_encoded, _ = encode_categorical_features_api(df)
    
    
    df_feature = feature_engineering(df_encoded)
    


    # Make sure we have exactly the columns the model expects
    df_final = ensure_expected_columns(df_feature, EXPECTED_COLUMNS)
    return df_final


# =========================
# 6) FastAPI app
# =========================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained Telco churn model.",
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {
        "message": "Churn model API is running",
        "docs": "/docs",
    }


# =========================
# 7) Prediction endpoint
# =========================

@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    """
    Receives RAW customer features → preprocesses them →
    returns churn prediction and probability.
    """
    # Convert input Pydantic model to DataFrame
    raw_df = pd.DataFrame([features.dict()])

    # Preprocess to match training-time features
    try:
        processed_df = preprocess_input(raw_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")

    # Predict
    try:
        y_pred = model.predict(processed_df)[0]
        prob: Optional[float] = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(processed_df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    pred_int = int(y_pred)
    label = "Churn" if pred_int == 1 else "No Churn"

    return {
        "churn_class": pred_int,
        "churn_label": label,
        "churn_probability": prob,
    }


# =========================
# 8) Optional: run with `python main.py`
# =========================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
