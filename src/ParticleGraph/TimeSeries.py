import os
from collections.abc import Sequence
from typing import Dict, List

import torch
from torch_geometric.data import Data

from ParticleGraph.field_descriptors import FieldDescriptor


class TimeSeries(Sequence):
    def __init__(
            self,
            time: torch.Tensor,
            data: Sequence[Data],
            field_descriptors: Dict[str, FieldDescriptor] = None,
    ):
        self.time = time
        self._data = data
        self.fields = field_descriptors

    def __len__(self) -> int:
        return len(self.time)

    def __getitem__(self, idx: int) -> Data:
        data = self._data[idx]
        return data

    @staticmethod
    def load(path: str) -> 'TimeSeries':
        try:
            fields = torch.load(os.path.join(path, 'fields.pt'))
            time = torch.load(os.path.join(path, 'time.pt'))

            n_time_steps = len(time)
            n_digits = len(str(n_time_steps - 1))

            data = []
            for i in range(n_time_steps):
                data.append(torch.load(os.path.join(path, f'data_{str(i).zfill(n_digits)}.pt')))
        except Exception as e:
            raise ValueError(f"Could not load data from {path}.") from e

        return TimeSeries(time, data, fields)

    @staticmethod
    def save(time_series: 'TimeSeries', path: str):
        try:
            os.makedirs(path, exist_ok=False)
            torch.save(time_series.fields, os.path.join(path, 'fields.pt'))
            torch.save(time_series.time, os.path.join(path, 'time.pt'))

            n_time_steps = len(time_series)
            n_digits = len(str(n_time_steps - 1))

            for i, d in enumerate(time_series):
                torch.save(d, os.path.join(path, f'data_{str(i).zfill(n_digits)}.pt'))
        except Exception as e:
            raise ValueError(f"Could not save data to {path}.") from e

    def compute_derivative(
            self,
            field_name: str,
            *,
            id_name: str = None
    ) -> List[torch.Tensor]:
        """
        Compute the backward difference quotient of a field in a time series.
        :param time_series: The time series over which to compute the difference quotient.
        :param field_name: The field for which to compute the difference quotient.
        :param id_name: If given, this field is used to match data points between time steps. Ids are assumed to be unique.
        :return: A list of tensors containing the difference quotient at each time step. Where the difference quotient could
            not be computed, the corresponding entry is Nan.
        """
        difference_quotients = [torch.full_like(getattr(self[0], field_name), torch.nan)]
        for i in range(1, len(self)):
            x_current = getattr(self[i], field_name)
            x_previous = getattr(self[i - 1], field_name)
            delta_t = self.time[i] - self.time[i - 1]

            if id_name is None:
                difference_quotients.append((x_current - x_previous) / delta_t)
            else:
                id_current = getattr(self[i], id_name)
                id_previous = getattr(self[i - 1], id_name)

                # Compute a set of global unique ids
                all_ids = torch.cat((id_current, id_previous))
                _, indices, counts = torch.unique(all_ids, return_inverse=True, return_counts=True)

                # Compute the difference quotient in the global id space
                indices_current = indices[:len(id_current)]
                indices_previous = indices[len(id_current):]
                all_differences = torch.bincount(indices_current, x_current, minlength=len(counts))
                all_differences -= torch.bincount(indices_previous, x_previous, minlength=len(counts))

                # Only consider ids that are present in both time steps and map to current ids
                all_differences[counts.ne(2)] = torch.nan
                difference_quotient = all_differences[indices_current] / delta_t
                difference_quotients.append(difference_quotient)

        return difference_quotients
