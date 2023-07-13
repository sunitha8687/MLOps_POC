import warnings
import numpy as np
warnings.filterwarnings('ignore', category=ResourceWarning)
import pickle


def inference(sensor_1,sensor_2):
    """
    This functions performs prediction by loading the pickled model.
    Parameters:
        sensor_1,sensor_2: latest inference data values
    Returns:
        ypred : predictions of the label
    """
    
    x_inference = np.array([sensor_1,sensor_2])
    x_inference = np.reshape(x_inference, (-1, 2))
    pickled_model = pickle.load(open('src/model.pkl', 'rb'))
    y_pred = pickled_model.predict_proba(x_inference)[:, 1]
    return y_pred
