import warnings
import numpy as np
warnings.filterwarnings('ignore', category=ResourceWarning)
import pickle


def inference(inference_df):
    """
    This functions performs prediction by loading the pickled model.
    Parameters:
        inference_df: latest inference data
    Returns:
        ypred : predictions of the label
    """

    X_inference = inference_df.values
    pickled_model = pickle.load(open('src/model.pkl', 'rb'))  # if running in local, add ../src two dots
    y_pred = pickled_model.predict_proba(X_inference)[:, 1]
    inferred_df = inference_df.assign(y_pred=np.round(y_pred, 0))
    return y_pred, inferred_df
