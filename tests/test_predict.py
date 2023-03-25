import pytest
import pandas as pd
from src.model_code_changes import prediction




@pytest.mark.predict

def test_predict(test_data):
    pred = prediction()
    inference_df = pd.read_csv(test_data)
    inference_df = inference_df[:500]
    f1_score, accuracy_score, precision, recall = pred.predict_evaluate(inference_df)
    # Check returned scores value
    rounded_f1 = round(f1_score, 2)
    assert rounded_f1 == 0.98
    # Check returned scores type
    assert isinstance(f1_score, float)