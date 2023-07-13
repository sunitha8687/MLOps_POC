from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def data_preprocess(df):
    """
        Data preprocessing is done in this function including train-test split.
        Parameters:
            df: Input dataframe
        Returns:
            x_train: Training data,
            x_test: Testing data,
            y_train: Training label,
            y_test: Testing label
            xx,yy: co-ordinates for plotting
    """

    X = df[['sensor_1', 'sensor_2']].values
    y = df[['label' ,]].values 
    X = StandardScaler().fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=99
    )
    h = 0.02  # meshgrid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    return x_train, x_test, y_train, y_test, xx, yy  


# Write test for preprocess - train, test data check length, its transformed scaled valzes are bet 0 to 1.- 