import torch
from ParticleGraph.GraphTimeSeries import GraphTimeSeries


def test_time_series_length():
    n = 10
    raw_data = [torch.rand((k, n)) for k in range(5, 10)]
    time_series = GraphTimeSeries(raw_data, feature_names=[f'feature_{i}' for i in range(n)])

    assert len(time_series) == len(raw_data)
