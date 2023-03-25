import pytest
import pandas as pd
from src.new_model import predict_evaluate




@pytest.mark.predict

def test_predict(test_data):
    inference_df = pd.read_csv(test_data)
    inference_df = inference_df[:500]
    f1_score, precision, recall = predict_evaluate(inference_df)
    # Check returned f1 score value
    rounded_f1 = round(f1_score, 2)
    assert rounded_f1 == 0.96
    # Check returned scores type
    assert isinstance(f1_score, float)