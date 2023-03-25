import pytest
import sys
from os.path import abspath
from os.path import dirname as d

@pytest.fixture()
def test_data():  # you cant download a model everytime, so need to put that in the fixture
    yield "data/historical_sensor_data.csv"  # add double dots ../data/ if running in local

data_science_problem = f"{d(d(abspath(__file__)))}"
sys.path.append(f"{data_science_problem}/src")