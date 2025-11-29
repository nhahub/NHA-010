# Customer Churn Prediction (NHA-010)

This repository contains an end-to-end machine learning project for **predicting customer churn**.  
It covers the full lifecycle from exploratory data analysis and feature engineering, to model training, evaluation, experiment tracking, and serving predictions via an API and simple frontend.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [How to Use](#how-to-use)
  - [1. Explore the Data (Notebooks)](#1-explore-the-data-notebooks)
  - [2. Train Models](#2-train-models)
  - [3. Evaluate Models](#3-evaluate-models)
  - [4. Serve Predictions via API](#4-serve-predictions-via-api)
  - [5. (Optional) Frontend](#5-optional-frontend)
- [Experiment Tracking](#experiment-tracking)
- [Reports](#reports)
- [Roadmap / Future Work](#roadmap--future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

Customer churn is a critical problem for subscription-based and service-oriented businesses.  
The goal of this project is to:

- Build and compare multiple ML models for **churn prediction** (e.g. Logistic Regression, Gradient Boosting, XGBoost).
- Identify the most important features driving churn.
- Provide reproducible pipelines for:
  - Data cleaning & feature engineering  
  - Model training & evaluation  
  - Serving predictions via an API and simple UI

---

## Features

- 📊 **Exploratory Data Analysis**  
  Jupyter notebooks for understanding distributions, correlations, and churn drivers.

- 🧱 **Modular ML Pipeline**  
  Separate modules for data loading, cleaning, feature engineering, model training, and evaluation.

- 🧪 **Multiple Models & Comparisons**  
  Includes several algorithms (e.g. Logistic Regression, Gradient Boosting, XGBoost), with metrics such as:
  - Accuracy, Precision, Recall, F1
  - ROC curves
  - Confusion matrices

- 📈 **Experiment Tracking**  
  Uses `mlruns/` (MLflow) to track runs, parameters, and metrics.

- 🌐 **API for Online Inference**  
  An `api/` service to expose the trained model for real-time predictions.

- 💻 **Frontend**  
  A `frontend/` directory for a simple UI to interact with the model (e.g. entering customer features and getting a churn probability).

---

## Repository Structure

> Adjust this section if your actual structure differs.
```
NHA-010/
├── api/                     # Backend API for model inference
├── data/
│   ├── raw/                 # Raw datasets (original, unmodified)
│   ├── processed/           # Cleaned / transformed datasets
│   └── external/            # Any external/extra datasets (optional, if used)
├── frontend/                # Frontend / UI for interacting with the model
├── mlruns/                  # MLflow experiment tracking artifacts
├── models/                  # Saved models, metrics, etc.
├── notebooks/               # Jupyter notebooks (EDA, feature engineering, training, evaluation)
├── reports/                 # Generated reports, figures, PDFs
├── src/
│   └── utils/               # Reusable utilities and helper functions
├── testing_artifacts/       # Artifacts generated during testing (if applicable)
├── Run_models.py            # Script for running / comparing models
├── main.py                  # Main execution script (training pipeline / entry point)
├── confusion_matrix*.png    # Confusion matrices for different models
├── roc_curve*.png           # ROC curves for different models
└── Readme.md                # Project documentation (this file)

```
## Tech Stack

**Language:**  
- Python 3.x  

**Data & ML:**  
- pandas  
- NumPy  
- scikit-learn  
- xgboost  

**Visualization:**  
- Matplotlib  
- Seaborn  

**Experiment Tracking:**  
- MLflow (`mlruns/`)  

**Serving (API):**  
- Flask / FastAPI (see `api/`)  

**Notebooks:**  
- Jupyter Notebook / JupyterLab  

*(Update this list to match your exact dependencies and versions.)*

---

## Getting Started

### Prerequisites
- Python 3.8+ (recommended)  
- git  
- (Optional) `virtualenv` or `conda` for environment management  


## Installation

### 1. Clone the repository
```bash
git clone https://github.com/nhahub/NHA-010.git
cd NHA-010
```

### 2. Create and activate a virtual environment (recommended)

```
python -m venv .venv
```
- On Linux / macOS:
```
source .venv/bin/activate
```

# How to Use

## 1. Explore the Data (Notebooks)

Open the notebooks to understand the workflow:  
Use `jupyter notebook notebooks/` to inspect:

- `01_data_exploration.ipynb`
- `02_feature_engineering.ipynb`
- `03_model_training.ipynb`
- `04_model_evaluation.ipynb`

These help you:

- Inspect distributions and correlations  
- Engineer features  
- Train and compare models  

---

## 2. Train Models

Run the model comparison script (`Run_models.py`) or the main training pipeline (`main.py`).

The training scripts will:

- Load and clean data (`data/raw/`)
- Perform feature engineering (`data/processed/`)
- Train Logistic Regression, Gradient Boosting, and XGBoost models
- Log metrics to `mlruns/` and save artifacts to `models/`

---

## 3. Evaluate Models

Evaluation outputs include:

- `confusion_matrix.png`
- `confusion_matrix_logreg.png`
- `confusion_matrix_gradient_boosting.png`
- `confusion_matrix_xgboost.png`
- `roc_curve*.png`

Additional evaluation and summaries can be found in:

- `notebooks/`
- `reports/`

---

## 4. Serve Predictions via API

FastAPI entry point: `uvicorn api.main:app --reload`  
Flask entry point: `python api/app.py`

### Example Prediction Request

POST `/predict` with JSON:

```json
{
  "tenure": 24,
  "monthly_charges": 75.5,
  "contract_type": "Month-to-month"
}
```

## 5. Frontend 

From the `frontend/` directory, open churn-predictor.html in your browser

Include UI documentation covering:

- How to start the UI  
- Required environment variables (API URL)  
- Entry URL: <[http://localhost:8000] >

---

## Experiment Tracking

MLflow runs are stored in the `mlruns/` directory.  
Start the MLflow UI with `mlflow ui --backend-store-uri mlruns` and open it at <http://127.0.0.1:5000>.

---

## Reports

The `reports/` directory may include:

- EDA reports  
- Model evaluation summaries  
- Final project reports  

---

## Roadmap / Future Work

- Improve hyperparameter tuning **(DONE)**  
- Add clearer documentation for API & frontend **(DONE)**  
- Add automated tests  
- Deploy model to cloud (Docker + provider)  
- Add monitoring & auto-retraining  

---

## Contributing

Typical workflow:

1. Create a feature branch  
2. Commit changes  
3. Push the branch  
4. Open a Pull Request  
