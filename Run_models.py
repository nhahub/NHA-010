import pandas as pd
from sklearn.model_selection import train_test_split

# models_mlflow.py
from src.utils.models_mlflow import (decision_tree_model_mlflow,logistic_regression_model_mlflow,xgboost_model_mlflow,gradient_boosting_model_mlflow)
import subprocess
import webbrowser
import time
import os

def start_mlflow_ui(port=5000):
    try:
      
        subprocess.Popen(
            ["mlflow", "ui", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )

        
        time.sleep(3)

      
        webbrowser.open(f"http://127.0.0.1:{port}")

    except Exception as e:
        print(f"Error starting MLflow UI: {e}")


# ========== 1. Load Dataset ==========
def load_data():
    df = pd.read_csv("data/extended_featured_data.csv")  
    return df

# ========== 2. Prepare Data ==========
def prepare_data(df):
 
    X = df.drop(columns = ['Churn','NEW_TENURE_YEAR'])      
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test

# ========== 3. Run Decision Tree Model with MLflow ==========
def run_decision_tree():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    decision_tree_model_mlflow(
        X_train, X_test, y_train, y_test,
        criterion="entropy",
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=None,
        random_state=99,
        experiment_name="DecisionTree Experiments",
        run_name="DecisionTree Run"       
    )

def run_logistic_regression():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    logistic_regression_model_mlflow(
        X_train, X_test, y_train, y_test,
        penalty="l1",
        C=0.5,
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced",
        random_state=10,
        experiment_name="Logistic Regression Experiments",
        run_name="LogisticRegression Run"
)




def run_xgboost_model():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    xgboost_model_mlflow(
        X_train, X_test, y_train, y_test,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.01,
        subsample=1.0,
        colsample_bytree=1.0,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42,
        experiment_name="XGBoost Experiments",
        run_name="XGBoost Run"
)



def run_gradient_boosting_model():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    gradient_boosting_model_mlflow(
    X_train, X_test, y_train, y_test,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=1.0,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    random_state=42,

    experiment_name="GradientBoosting Experiments",
    run_name="GradientBoosting Run"
)







if __name__ == "__main__":
    start_mlflow_ui()  
    time.sleep(2)

    run_decision_tree()
    run_logistic_regression()
    run_xgboost_model()
    run_gradient_boosting_model()