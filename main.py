from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib

# =========================
# 1) Load the trained model
# =========================

try:
    model = joblib.load("churn_model.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading churn_model.pkl: {e}")


# =========================
# 2) Input schema (RAW data)
#    Same columns as the original Telco dataset
#    except: we exclude customerID and Churn
# =========================

class CustomerFeatures(BaseModel):
    gender: str                  # "Male" / "Female"
    SeniorCitizen: int           # 0 / 1
    Partner: str                 # "Yes" / "No"
    Dependents: str              # "Yes" / "No"
    tenure: int
    PhoneService: str            # "Yes" / "No"
    MultipleLines: str           # "Yes" / "No" / "No phone service"
    InternetService: str         # "DSL" / "Fiber optic" / "No"
    OnlineSecurity: str          # "Yes" / "No" / "No internet service"
    OnlineBackup: str            # "Yes" / "No" / "No internet service"
    DeviceProtection: str        # "Yes" / "No" / "No internet service"
    TechSupport: str             # "Yes" / "No" / "No internet service"
    StreamingTV: str             # "Yes" / "No" / "No internet service"
    StreamingMovies: str         # "Yes" / "No" / "No internet service"
    Contract: str                # "Month-to-month" / "One year" / "Two year"
    PaperlessBilling: str        # "Yes" / "No"
    PaymentMethod: str           # "Electronic check" / "Mailed check" / "Bank transfer (automatic)" / "Credit card (automatic)"
    MonthlyCharges: float
    TotalCharges: float          # might be string in original CSV, but we treat as numeric here


# =========================
# 3) Expected final feature columns (after preprocessing)
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
    "NEW_AVG_Service_Fee"
]


# =========================
# 4) Preprocessing:
#    RAW → engineered numeric features
#    Must match how you trained the model
# =========================

def preprocess_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert one row of RAW Telco customer data into
    the 37 engineered features used to train churn_model.pkl.

    IMPORTANT:
    If you defined NEW_* features differently in your notebook,
    update this logic to match your notebook exactly.
    """

    df = raw_df.copy()

    # ---------- Basic cleaning / encoding ----------

    # gender: Male=1, Female=0 (change if you used opposite)
    df["gender"] = df["gender"].map({"Female": 0, "Male": 1}).fillna(0).astype(int)

    # Yes/No → 1/0
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df[col] = df[col].map({"No": 0, "Yes": 1}).fillna(0).astype(int)

    # SeniorCitizen: ensure integer
    df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    # TotalCharges: ensure numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # MonthlyCharges should already be float
    df["MonthlyCharges"] = df["MonthlyCharges"].fillna(0.0).astype(float)

    # ---------- Manual one-hot-like encoding to match your final column names ----------

    # MultipleLines
    df["MultipleLines_No phone service"] = (df["MultipleLines"] == "No phone service").astype(int)
    df["MultipleLines_Yes"] = (df["MultipleLines"] == "Yes").astype(int)

    # InternetService
    df["InternetService_Fiber optic"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["InternetService_No"] = (df["InternetService"] == "No").astype(int)
    # DSL is the base case

    # OnlineSecurity
    df["OnlineSecurity_No internet service"] = (df["OnlineSecurity"] == "No internet service").astype(int)
    df["OnlineSecurity_Yes"] = (df["OnlineSecurity"] == "Yes").astype(int)

    # OnlineBackup
    df["OnlineBackup_No internet service"] = (df["OnlineBackup"] == "No internet service").astype(int)
    df["OnlineBackup_Yes"] = (df["OnlineBackup"] == "Yes").astype(int)

    # DeviceProtection
    df["DeviceProtection_No internet service"] = (df["DeviceProtection"] == "No internet service").astype(int)
    df["DeviceProtection_Yes"] = (df["DeviceProtection"] == "Yes").astype(int)

    # TechSupport
    df["TechSupport_No internet service"] = (df["TechSupport"] == "No internet service").astype(int)
    df["TechSupport_Yes"] = (df["TechSupport"] == "Yes").astype(int)

    # StreamingTV
    df["StreamingTV_No internet service"] = (df["StreamingTV"] == "No internet service").astype(int)
    df["StreamingTV_Yes"] = (df["StreamingTV"] == "Yes").astype(int)

    # StreamingMovies
    df["StreamingMovies_No internet service"] = (df["StreamingMovies"] == "No internet service").astype(int)
    df["StreamingMovies_Yes"] = (df["StreamingMovies"] == "Yes").astype(int)

    # Contract
    df["Contract_One year"] = (df["Contract"] == "One year").astype(int)
    df["Contract_Two year"] = (df["Contract"] == "Two year").astype(int)
    # "Month-to-month" is the base

    # PaymentMethod
    df["PaymentMethod_Credit card (automatic)"] = (df["PaymentMethod"] == "Credit card (automatic)").astype(int)
    df["PaymentMethod_Electronic check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["PaymentMethod_Mailed check"] = (df["PaymentMethod"] == "Mailed check").astype(int)
    # "Bank transfer (automatic)" is the base

    # ---------- Engineered features (example definitions) ----------
    # NOTE: Adjust these if your notebook used different logic.

    # 1) NEW_noProt:
    # no OnlineSecurity, no DeviceProtection, and no TechSupport
    df["NEW_noProt"] = (
        ((df["OnlineSecurity"] == "No") | (df["OnlineSecurity"] == "No internet service")) &
        ((df["DeviceProtection"] == "No") | (df["DeviceProtection"] == "No internet service")) &
        ((df["TechSupport"] == "No") | (df["TechSupport"] == "No internet service"))
    ).astype(int)

    # 2) NEW_Engaged:
    # Contract of at least one year OR tenure >= 12 months
    df["NEW_Engaged"] = (
        (df["Contract"].isin(["One year", "Two year"])) | (df["tenure"] >= 12)
    ).astype(int)

    # 3) NEW_Young_Not_Engaged:
    # low tenure and month-to-month
    df["NEW_Young_Not_Engaged"] = (
        (df["tenure"] <= 12) & (df["Contract"] == "Month-to-month")
    ).astype(int)

    # 4) NEW_FLAG_ANY_STREAMING:
    # has any streaming service
    df["NEW_FLAG_ANY_STREAMING"] = (
        (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
    ).astype(int)

    # 5) NEW_FLAG_AutoPayment:
    # automatic payment methods
    df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)

    # 6) NEW_TotalServices:
    # count how many services are active (approximation, adjust if needed)
    service_conditions = [
        (df["PhoneService"] == 1),
        (df["MultipleLines"] == "Yes"),
        (df["InternetService"] != "No"),
        (df["OnlineSecurity"] == "Yes"),
        (df["OnlineBackup"] == "Yes"),
        (df["DeviceProtection"] == "Yes"),
        (df["TechSupport"] == "Yes"),
        (df["StreamingTV"] == "Yes"),
        (df["StreamingMovies"] == "Yes"),
    ]
    df["NEW_TotalServices"] = sum(cond.astype(int) for cond in service_conditions)

    # 7) NEW_AVG_Service_Fee:
    # average monthly charge per service
    df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / df["NEW_TotalServices"].replace(0, 1)
    df["NEW_AVG_Service_Fee"] = df["NEW_AVG_Service_Fee"].astype(float)

    # ---------- Keep only EXPECTED_COLUMNS in the right order ----------
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            # if any expected column is missing, fill with 0
            df[col] = 0

    df = df[EXPECTED_COLUMNS]

    return df


# =========================
# 5) FastAPI app
# =========================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained Telco churn model.",
    version="1.0.0"
)


@app.get("/")
def read_root():
    return {
        "message": "Churn model API is running",
        "docs": "/docs"
    }


# =========================
# 6) Prediction endpoint
# =========================

@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    """
    Receives RAW customer features → preprocesses them →
    returns churn prediction and probability.
    """

    # Convert input to DataFrame with one row
    raw_df = pd.DataFrame([features.dict()])

    # Preprocess to match training features
    processed_df = preprocess_input(raw_df)

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
        "churn_probability": prob
    }
