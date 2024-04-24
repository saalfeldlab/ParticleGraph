import os
from collections.abc import Sequence

import torch
from torch_geometric.data import Data


class TimeSeries(Sequence):
    def __init__(
            self,
            time: torch.Tensor,
            data: Sequence[Data],
    ):
        self.time = time
        self._data = data

    def __len__(self) -> int:
        return len(self.time)

    def __getitem__(self, idx: int) -> Data:
        data = self._data[idx]
        return data

    @staticmethod
    def load(path: str) -> 'TimeSeries':
        try:
            time = torch.load(os.path.join(path, 'time.pt'))

            n_time_steps = len(time)
            n_digits = len(str(n_time_steps - 1))

            data = []
            for i in range(n_time_steps):
                data.append(torch.load(os.path.join(path, f'data_{str(i).zfill(n_digits)}.pt')))
        except Exception as e:
            raise ValueError(f"Could not load data from {path}.") from e

        return TimeSeries(time, data)

    @staticmethod
    def save(time_series: 'TimeSeries', path: str):
        try:
            os.makedirs(path, exist_ok=False)
            torch.save(time_series.time, os.path.join(path, 'time.pt'))

            n_time_steps = len(time_series)
            n_digits = len(str(n_time_steps - 1))

            for i, d in enumerate(time_series):
                torch.save(d, os.path.join(path, f'data_{str(i).zfill(n_digits)}.pt'))
        except Exception as e:
            raise ValueError(f"Could not save data to {path}.") from e
