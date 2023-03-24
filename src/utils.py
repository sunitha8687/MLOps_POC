import mlflow.sklearn
from model_code_changes import *


# Note: on Databricks, the experiment name passed to mlflow_set_experiment must be a
# valid path in the workspace

remote_server_uri = "http://localhost:5000"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment("/Classification model ex1")
mlflow.log_param('kernel', 1.0 * RBF(1.0))
mlflow.log_metric("f1-score", score)
mlflow.set_tag("Model", "Gaussian Process classifier")
mlflow.sklearn.log_model(clf, "model")
mlflow.log_artifact("src/Raw input data.png")
mlflow.log_artifact("src/Classifier output.png")
mlflow.log_artifact("src/Classifier inference output.png")
mlflow.log_artifact("src/data/historical_sensor_data.csv")
mlflow.log_artifact("src/data/latest_sensor_data.csv")


