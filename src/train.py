import warnings
import numpy as np
warnings.filterwarnings('ignore', category=DeprecationWarning)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
warnings.filterwarnings('ignore', category=ResourceWarning)
import pickle


def train_model(x_train, x_test, y_train, y_test, xx, yy):
    """
    This functions trains a classification model and plots the training and testing data along with the
    labels to visualize the classifier output.
    Parameters:
        X_train: Training data,
        X_test: Testing data,
        y_train: Training label,
        y_test: Testing label
        xx,yy: co-ordinates for plotting
    Returns:
        clf: classifying model
        score: score of the test data
    """

    kernel = 1.0 * RBF(1.0)
    clf = GaussianProcessClassifier(kernel=kernel)
    clf.fit(x_train, y_train.ravel())
    score = clf.score(x_test, y_test)
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    pickle.dump(clf, open('model.pkl', 'wb'))

    return clf, score