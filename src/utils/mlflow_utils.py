import mlflow
from mlflow.tracking import MlflowClient

def get_or_create_experiment(name="testing models"):
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)

    if exp:
        return exp.experiment_id
    else:
        return client.create_experiment(
            name=name,
            artifact_location="testing_artifacts",
            tags={"env": "dev", "version": "1.0.0"}
        )


def start_experiment_run(experiment_id, run_name=None):
    return mlflow.start_run(experiment_id=experiment_id, run_name=run_name)


def log_params(params: dict):
    for k, v in params.items():
        mlflow.log_param(k, v)


def log_metrics(metrics: dict):
    for k, v in metrics.items():
        mlflow.log_metric(k, v)







