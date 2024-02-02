from torch_geometric.data import Data
from collections.abc import Iterable

class GraphTimeSeries:
    """
    A class that has a list of torch_geometric.data.Data objects and a feature name map
    """

    _data: list[Data]
    _feature_index: dict[str, int]

    def __init__(self, data_list, *, feature_names, time_points=None):
        self._data = data_list
        self._feature_index = {feature: i for i, feature in enumerate(feature_names)}
        if time_points is not None:
            self.time_points = time_points
        else:
            self.time_points = list(range(len(data_list)))

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)

    def get_feature_names(self):
        return self._feature_index.keys()

    def get_indices_for(self, feature_name):
        if isinstance(feature_name, Iterable):
            return [self._feature_index[feature] for feature in feature_name]
        else:
            return self._feature_index[feature_name]