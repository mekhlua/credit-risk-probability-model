import pandas as pd
# At the very top of tests/test_data_processing.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_processing import compute_rfm, assign_high_risk

def test_compute_rfm():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'TransactionStartTime': ['2023-01-01', '2023-01-10', '2023-01-05'],
        'TransactionId': [101, 102, 103],
        'Amount': [100, 200, 300]
    })
    snapshot_date = pd.to_datetime('2023-01-15')
    rfm = compute_rfm(df, snapshot_date)
    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns

def test_assign_high_risk():
    rfm = pd.DataFrame({
        'Recency': [10, 5, 20],
        'Frequency': [2, 5, 1],
        'Monetary': [300, 500, 100]
    }, index=[1,2,3])
    result = assign_high_risk(rfm)
    assert 'is_high_risk' in result.columns
    assert set(result['is_high_risk'].unique()).issubset({0,1})