import pytest
import torch
from ParticleGraph.GraphTimeSeries import GraphTimeSeries

@pytest.fixture
def time_series():
    n = 5
    raw_data = [torch.rand((k, n)) for k in range(n, n+5)]
    feature_names = [f'feature_{i}' for i in range(n)]
    return GraphTimeSeries(raw_data, feature_names=feature_names)

def test_time_series_length(time_series):
    assert len(time_series) == 5

def test_time_series_get_item(time_series):
    assert time_series[3].shape == (8, 5)

def test_feature_indexing(time_series):
    assert time_series.get_feature_names() == ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
    assert time_series.get_indices_for('feature_3') == 3
    assert time_series.get_indices_for(['feature_3', 'feature_4']) == [3, 4]