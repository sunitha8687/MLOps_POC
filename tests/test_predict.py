import pytest
import pandas as pd
from src.new_model import predict_evaluate


@pytest.mark.predict
def test_predict(test_data):
    inference_df = pd.read_csv(test_data)
    inference_df = inference_df[:500]
    model, f1_score, precision, recall = predict_evaluate(inference_df)
    # Check returned scores type
    assert precision >= 0.9
    assert recall >= 0.9
    assert f1_score >= 0.9
    assert isinstance(f1_score, float)

    #pytest -m predict -- integrate this in ci cd. 
    #integration test - to check the flask api. 