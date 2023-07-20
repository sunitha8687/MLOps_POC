from fastapi.testclient import TestClient
from src.app import app
import pytest
import httpx

@pytest.fixture
def client():
    client = TestClient(app)
    yield client

@pytest.mark.integration
def test_predict_endpoint(client):
   # test data
   payload =  {
   "sensor_1": "1.6623607876951856",
   "sensor_2": "0.465600727640637"
   }

   # Send request to the Fast API
   response = client.post("/infer", json = payload)

   # Perform assertions on the response
   assert response.status_code == 200
   assert isinstance(response.json()['predictions'], str)


