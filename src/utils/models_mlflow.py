import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score
from mlflow.models.signature import infer_signature
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



from src.utils.mlflow_utils import (
    get_or_create_experiment,
    start_experiment_run,
    log_params,
    log_metrics
)

def decision_tree_model_mlflow(
    X_train, X_test, y_train, y_test,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    random_state=42,
    experiment_name="DecisionTree Experiments",
    run_name="DecisionTree Run"
):


    # 1) Create / Get experiment
    experiment_id = get_or_create_experiment(experiment_name)

    # 2) Start run
    with start_experiment_run(experiment_id, run_name):

        # ---- Train model ----
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ---- Metrics ----
        f1_score3=f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # ---- Log parameters ----
        params = {
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state
        }
        log_params(params)

        # ---- Log accuracy ----
        log_metrics({"accuracy": accuracy,"f1_score":f1_score3})

        # ---------- CONFUSION MATRIX FIGURE ----------
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        # Save figure temporarily
        fig_path = "confusion_matrix.png"
        plt.savefig(fig_path)
        plt.close()

        # Log to MLflow
        mlflow.log_artifact(fig_path)

        # Log the model in MLflow
        signature = infer_signature(X_train.astype("float64"), model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="decision_tree_model",
            signature=signature,
            input_example=X_train.iloc[:5].astype("float64")
            )

        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        else:
            # fallback (if model does not support predict_proba)
            y_probs = model.decision_function(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

        roc_path = "roc_curve.png"
        plt.savefig(roc_path)
        plt.close()

        # Log ROC curve in MLflow
        mlflow.log_artifact(roc_path)

        # Log AUC metric
        log_metrics({"AUC": roc_auc})
        # ----------------------------------------------

    return model, accuracy, y_pred,f1_score3




def logistic_regression_model_mlflow(
    X_train, X_test, y_train, y_test,
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    class_weight=None,
    random_state=42,
    experiment_name="Logistic Regression Experiments",
    run_name="LogisticRegression Run"
):

    # 1) Create / Get experiment
    experiment_id = get_or_create_experiment(experiment_name)

    # 2) Start run
    with start_experiment_run(experiment_id, run_name):

        # ---- Train model ----
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ---- Metrics ----
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # ---- Log parameters ----
        params = {
            "penalty": penalty,
            "C": C,
            "solver": solver,
            "max_iter": max_iter,
            "class_weight": class_weight,
            "random_state": random_state
        }
        log_params(params)

        # ---- Log metrics ----
        log_metrics({"accuracy": accuracy, "f1_score": f1})

        # ---------- CONFUSION MATRIX FIGURE ----------
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        fig_path = "confusion_matrix_logreg.png"
        plt.savefig(fig_path)
        plt.close()

        mlflow.log_artifact(fig_path)

        # ---------- Model Signature ----------
        signature = infer_signature(
            X_train.astype("float64"),
            model.predict(X_train)
        )

        # ---------- Log Model ----------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="logistic_regression_model",
            signature=signature,
            input_example=X_train.iloc[:5].astype("float64")
        )
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        else:
            # fallback (if model does not support predict_proba)
            y_probs = model.decision_function(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

        roc_path = "roc_curve.png"
        plt.savefig(roc_path)
        plt.close()

        # Log ROC curve in MLflow
        mlflow.log_artifact(roc_path)

        # Log AUC metric
        log_metrics({"AUC": roc_auc})



    return model, accuracy, y_pred, f1





def xgboost_model_mlflow(
    X_train, X_test, y_train, y_test,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    random_state=42,
    experiment_name="XGBoost Experiments",
    run_name="XGBoost Run"
):

    from xgboost import XGBClassifier

    # 1) Create / Get experiment
    experiment_id = get_or_create_experiment(experiment_name)

    # 2) Start run
    with start_experiment_run(experiment_id, run_name):

        # ---- Train model ----
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            eval_metric="logloss",
            use_label_encoder=False
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ---- Metrics ----
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # ---- Log parameters ----
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state
        }
        log_params(params)

        # ---- Log metrics ----
        log_metrics({"accuracy": accuracy, "f1_score": f1})

        # ---------- CONFUSION MATRIX FIGURE ----------
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        fig_path = "confusion_matrix_xgboost.png"
        plt.savefig(fig_path)
        plt.close()

        mlflow.log_artifact(fig_path)

        # ---------- Model Signature ----------
        signature = infer_signature(
            X_train.astype("float64"),
            model.predict(X_train)
        )

        # ---------- Log Model ----------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="xgboost_model",
            signature=signature,
            input_example=X_train.iloc[:5].astype("float64")
        )

        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        else:
            # fallback (if model does not support predict_proba)
            y_probs = model.decision_function(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

        roc_path = "roc_curve.png"
        plt.savefig(roc_path)
        plt.close()

        # Log ROC curve in MLflow
        mlflow.log_artifact(roc_path)

        # Log AUC metric
        log_metrics({"AUC": roc_auc})



    return model, accuracy, y_pred, f1





def gradient_boosting_model_mlflow(
    X_train, X_test, y_train, y_test,

    # -------- Gradient Boosting Parameters --------
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
):

    from sklearn.ensemble import GradientBoostingClassifier

    # 1) Create / Get experiment
    experiment_id = get_or_create_experiment(experiment_name)

    # 2) Start run
    with start_experiment_run(experiment_id, run_name):

        # ---- Train model ----
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ---- Metrics ----
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # ---- Log parameters ----
        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state
        }
        log_params(params)

        # ---- Log metrics ----
        log_metrics({"accuracy": accuracy, "f1_score": f1})

        # ---------- CONFUSION MATRIX FIGURE ----------
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        fig_path = "confusion_matrix_gradient_boosting.png"
        plt.savefig(fig_path)
        plt.close()

        mlflow.log_artifact(fig_path)

        # ---------- Model Signature ----------
        signature = infer_signature(
            X_train.astype("float64"),
            model.predict(X_train)
        )

        # ---------- Log Model ----------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="gradient_boosting_model",
            signature=signature,
            input_example=X_train.iloc[:5].astype("float64")
        )

        # ---------- ROC & AUC ----------
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        else:
            y_probs = model.decision_function(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

        roc_path = "roc_curve_gradient_boosting.png"
        plt.savefig(roc_path)
        plt.close()

        mlflow.log_artifact(roc_path)
        log_metrics({"AUC": roc_auc})

    return model, accuracy, y_pred, f1

