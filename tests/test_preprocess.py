import pytest
import pandas as pd
import numpy as np
from src.preprocess import data_quality_checks

@pytest.mark.preprocess
def test_data_quality_checks(test_data):
    raw_data = pd.read_csv(test_data)
    list_columns = raw_data.columns.tolist()
    column_types =  raw_data.dtypes.to_dict()
    duplicated_df = raw_data[raw_data.duplicated()]
    #column_names, column_types, duplicated_df = data_quality_checks(raw_data)
    data_schema_colummns = ['sensor_1','sensor_2','label']
    data_schema_types = {'sensor_1': np.dtype('float64'),
                         'sensor_2':np.dtype('float64'),
                         'label':np.dtype('float64')}
    assert data_schema_colummns == list_columns
    assert data_schema_types == column_types
    assert duplicated_df.empty

def test_preprocess():
    pass
