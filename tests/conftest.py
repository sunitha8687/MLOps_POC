import pytest

@pytest.fixture()
def test_data(): # you cant download a model everytime, so need to put that in the fixture
    yield "data/historical_sensor_data.csv"