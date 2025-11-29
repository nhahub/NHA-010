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

```text
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
