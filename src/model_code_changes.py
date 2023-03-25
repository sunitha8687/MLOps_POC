#!/usr/bin/env python
# coding: utf-8

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
warnings.filterwarnings('ignore', category=DeprecationWarning)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


class prediction():
    """
    Class to setup and perform model training
    """

    def data_preprocess(self,df):
        """
            Data preprocessing is done in this function including train-test split.
            Parameters:
                df: Input dataframe
            Returns:
                X_train: Training data,
                X_test: Testing data,
                y_train: Training label,
                y_test: Testing label
                xx,yy: co-ordinates for plotting
        """
        X = df[['sensor_1', 'sensor_2']].values
        y = df[['label',]].values
        X = StandardScaler().fit_transform(X)
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=99
        )
        h = 0.02 # meshgrid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        return x_train, x_test, y_train, y_test, xx, yy

    def data_visualize(self, x_train, x_test, xx, yy):
        """
        This function is used to visualize the raw input data by plotting data points in scatter plots.
        Parameters:
            X_train: Training data,
            X_test: Testing data,
            xx,yy: co-ordinates for plotting
        Returns:
            plt: scatter plot of training and testing data except the labels.
        """
        plt.figure(figsize=(10,8))
        cm = plt.cm.PiYG
        cm_bright = ListedColormap(["#FF0000", "#00ff5e"])
        plt.title("Input data")

        # Plot the training points
        plt.scatter(
            x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # Plot the testing points
        plt.scatter(
            x_test[:, 0], x_test[:, 1], c=y_test, marker='x',  cmap=cm_bright, alpha=1
        )
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.savefig("src/Raw input data.png")

        return plt, cm, cm_bright

    def train_model(self, x_train, x_test, y_train, y_test, xx, yy, cm, cm_bright):
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
        """

        kernel = 1.0 * RBF(1.0)
        clf = GaussianProcessClassifier(kernel=kernel)
        clf.fit(x_train, y_train.ravel())
        score = clf.score(x_test, y_test)
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(10,10))
        ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

        # Plot the training points
        ax.scatter(
            x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # Plot the testing points
        ax.scatter(
            x_test[:, 0],
            x_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title("Classifier output")
        ax.text(
            xx.max() - 0.3,
            yy.min() + 0.3,
            (f"score = {score:.2f}").lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        ax.figure.savefig("src/Classifier output.png")

        return clf, xx, yy, Z, score

    def predict_probab(self, inference_df, cm, cm_bright, xx, yy, Z):
        """
        This functions predicts the label using the guassian classifier model and plots the output of the inference data
        Parameters:
            inference_df: latest inference data
            cm,cm_bright: colour map in plots
            xx,yy: co-ordinates for plotting
            z: reshaping the predicted probability
        Returns:
            ypred : predictions of the label
        """

        X_inference = inference_df.values
        # predict labels
        y_pred = clf.predict_proba(X_inference)[:, 1]
        inference_df = inference_df.assign(y_pred= np.round(y_pred, 0))

        fig, ax = plt.subplots(figsize=(10,10))
        ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

        # Plot the inference points
        ax.scatter(
            X_inference[:, 0], X_inference[:, 1], marker="x", c=y_pred, cmap=cm_bright
        )
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title("Classifier inference output")
        ax.figure.savefig("src/Classifier inference output.png")

        return y_pred

    def predict_evaluate(self, df):
        x_values = df[['sensor_1', 'sensor_2']]
        y_values = df['label']

        x_train_values, x_test_values, y_train_values, y_test_values = train_test_split(
            x_values, y_values, test_size=0.3, random_state=99
        )
        model = GaussianProcessClassifier(1.0 * RBF(1.0))
        model.fit(x_train_values, y_train_values)
        y_pred_values = model.predict(x_test_values)
        f1_score = metrics.f1_score(y_test_values, y_pred_values)
        accuracy_score = metrics.accuracy_score(y_test_values, y_pred_values)
        precision = metrics.precision_score(y_test_values, y_pred_values)
        recall = metrics.recall_score(y_test_values, y_pred_values)
        return f1_score, accuracy_score, precision, recall

if __name__ == '__main__':

    try:
        df = pd.read_csv("src/data/historical_sensor_data.csv", sep=',')
        inference_df = pd.read_csv("src/data/latest_sensor_data.csv", sep=',')
    except Exception as e:
        logger.exception("Unable to download training and inference CSV, check your internet connection. Error: %s",
                         e)

    get_prediction = prediction()
    x_train, x_test, y_train, y_test, xx, yy = get_prediction.data_preprocess(df)
    plt, cm, cm_bright = get_prediction.data_visualize(x_train, x_test, xx, yy)
    clf, xx, yy, Z, score = get_prediction.train_model(x_train, x_test, y_train, y_test, xx, yy, cm, cm_bright)
    pred_results = get_prediction.predict_probab(inference_df, cm, cm_bright, xx, yy, Z)
    f1_score, accuracy_score, precision, recall = get_prediction.predict_evaluate(df)




    

