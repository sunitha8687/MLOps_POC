from fastapi import FastAPI
from src.predict import inference
from pydantic import BaseModel


app = FastAPI()

class ModelInput(BaseModel):
    sensor_1: float
    sensor_2: float

@app.post('/infer')
def run_inference(input_to_model:ModelInput):
    json_input = input_to_model.dict()
    y_pred = inference(json_input["sensor_1"],json_input["sensor_2"])
    if y_pred > 0.9:
        return {"predictions":"Non-faulty"}
    else:
        return {"predictions":"Faulty"} 