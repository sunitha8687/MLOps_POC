import mlflow.sklearn
import pandas as pd
import logging
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from sklearn.gaussian_process.kernels import RBF
from preprocess import data_preprocess  # if running in local: include src.preprocess
from train import train_model
from predict import inference
from new_model import predict_evaluate


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# Note: on Databricks, the experiment name passed to mlflow_set_experiment must be a valid path in the workspace
# The below function of model logging using mlflow is used only for local testing and not for CI in this case
def model_logging():
    """
    This function logs the model, params, metrics, artifacts to mlflow tracking server.
    """
    remote_server_uri = "http://localhost:5000"  # set to your server URI
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment("/Classification model ex1")
    mlflow.log_param('kernel', 1.0 * RBF(1.0))
    mlflow.log_metric("f1-score", score)
    mlflow.set_tag("Model", "Gaussian Process classifier")
    mlflow.sklearn.log_model(clf, "model")
    mlflow.log_artifact("./Raw input data.png")
    mlflow.log_artifact("./Classifier output.png")
    mlflow.log_artifact("./Classifier inference output.png")
    mlflow.log_artifact("./data/historical_sensor_data.csv")
    mlflow.log_artifact("./data/latest_sensor_data.csv")


if __name__ == '__main__':

    try:
        df = pd.read_csv("data/historical_sensor_data.csv", sep=',')  # add double dots ../data/ if running in local
        inference_df = pd.read_csv("data/latest_sensor_data.csv", sep=',')
    except Exception as e:
        logger.exception("Unable to download training and inference CSV, check your internet connection. Error: %s", e)

    x_train, x_test, y_train, y_test, xx, yy = data_preprocess(df)
    clf, score = train_model(x_train, x_test, y_train, y_test, xx, yy)
    pred_results,inferred_df = inference(inference_df)
    model, f1_score, precision, recall = predict_evaluate(df)
    # model_logging() # uncomment when running in local. Atm, not prepared for CI/CD only in this case.



