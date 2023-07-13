import pytest
import pandas as pd
import numpy as np
from src.preprocess import data_preprocess


@pytest.mark.preprocess
def test_data_quality_checks(test_data):
    raw_data = pd.read_csv(test_data)
    list_columns = raw_data.columns.tolist()
    column_types =  raw_data.dtypes.to_dict()
    duplicated_df = raw_data[raw_data.duplicated()]

    data_schema_colummns = ['sensor_1','sensor_2','label']
    data_schema_types = {'sensor_1': np.dtype('float64'),
                         'sensor_2':np.dtype('float64'),
                         'label':np.dtype('float64')}
    assert data_schema_colummns == list_columns
    assert data_schema_types == column_types
    assert duplicated_df.empty

@pytest.mark.evaluate
def test_preprocess(test_data):
    raw_data = pd.read_csv(test_data)
    raw_data = raw_data[:500]
    x_train, x_test, y_train, y_test, xx, yy = data_preprocess(raw_data)
    concat_arr = np.concatenate([x_train, x_test])
    np.unique(concat_arr)
    assert concat_arr.shape[0] == x_train.shape[0] + x_test.shape[0]

