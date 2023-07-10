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
        inferred_df: df with predictions
    """

    x_inference = inference_df.values
    pickled_model = pickle.load(open('src/model.pkl', 'rb'))
    y_pred = pickled_model.predict_proba(x_inference)[:, 1]
    inferred_df = inference_df.assign(y_pred=np.round(y_pred, 0))
    return y_pred, inferred_df
