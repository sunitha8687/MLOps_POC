import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split


def predict_evaluate(df):
    x_values = df[['sensor_1', 'sensor_2']]
    y_values = df['label']

    x_train_values, x_test_values, y_train_values, y_test_values = train_test_split(
        x_values, y_values, test_size=0.3, random_state=99
    )
    model = svm.SVC(kernel='rbf')
    model.fit(x_train_values, y_train_values)
    y_pred_values = model.predict(x_test_values)
    f1_score = metrics.f1_score(y_test_values, y_pred_values)
    precision = metrics.precision_score(y_test_values, y_pred_values)
    recall = metrics.recall_score(y_test_values, y_pred_values)
    return model, f1_score, precision, recall