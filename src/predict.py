import warnings
import numpy as np
warnings.filterwarnings('ignore', category=ResourceWarning)
import pickle

def predict_probabilities(inference_df):
    """
    This functions predicts the label using the guassian classifier model and plots the output of the inference data
    Parameters:
        inference_df: latest inference data
    Returns:
        ypred : predictions of the label
    """

    X_inference = inference_df.values
    # predict labels
    pickled_model = pickle.load(open('model.pkl', 'rb'))
    y_pred = pickled_model.predict_proba(X_inference)[:, 1]
    inference_df = inference_df.assign(y_pred=np.round(y_pred, 0))

    return y_pred
