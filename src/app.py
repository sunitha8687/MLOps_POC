from fastapi import FastAPI
from src.predict import inference
import pandas as pd
from pydantic import BaseModel


app = FastAPI()

class ModelInput(BaseModel):
    sensor_1: float
    sensor_2: float

@app.post('/infer')
def run_inference(input_to_model:ModelInput):
    json_input = input_to_model.dict()
    inference_df = pd.DataFrame.from_dict(json_input, orient='index')
    y_pred, inferred_df = inference(inference_df)
    if inferred_df['y_pred'] == 1.0:
        predictions = "Non-faulty"
    else:
        predictions = "Faulty"
    return predictions


# import client from pytest to do integration tests. we need a fixture. 
# Send the post request as input and write assert statements. 