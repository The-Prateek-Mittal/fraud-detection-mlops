import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import pytest
from src.train import preprocess

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Time': [0, 3600, 7200, 86400],
        'Amount': [10.0, 50.0, 100.0, 0.0],
        'V1': [0.1, 0.2, 0.3, 0.4]
    })

def test_preprocess_adds_amount_log(sample_data):
    df = preprocess(sample_data.copy())
    assert 'Amount_log' in df.columns
    expected_log = np.log1p(sample_data['Amount'])
    pd.testing.assert_series_equal(df['Amount_log'], expected_log, check_names=False)

def test_preprocess_adds_hour(sample_data):
    df = preprocess(sample_data.copy())
    assert 'Hour' in df.columns
    # 0 -> 0, 3600 -> 1, 7200 -> 2, 86400 -> 0
    assert list(df['Hour']) == [0, 1, 2, 0]

def test_preprocess_drops_time_amount(sample_data):
    df = preprocess(sample_data.copy())
    assert 'Time' not in df.columns
    assert 'Amount' not in df.columns

def test_hour_values_between_0_and_23(sample_data):
    # Add some random times
    large_data = pd.DataFrame({
        'Time': np.random.randint(0, 1000000, 100),
        'Amount': np.random.rand(100) * 100
    })
    df = preprocess(large_data)
    assert df['Hour'].min() >= 0
    assert df['Hour'].max() <= 23
