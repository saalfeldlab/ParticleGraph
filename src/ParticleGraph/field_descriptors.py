import abc
from dataclasses import dataclass
from typing import List

import numpy as np
from astropy.units import Unit


class FieldDescriptor(abc.ABC):
    """A class to describe the origin of a field in a dataset."""


@dataclass
class CsvDescriptor(FieldDescriptor):
    """A class to describe the origin of a field in a dataset as a column of a CSV file."""
    filename: str
    column_name: str
    type: np.dtype
    unit: Unit

    def __str__(self):
        return f"Column '{self.column_name}' from '{self.filename}' as {self.type} with unit '{self.unit.__repr__()}'."


@dataclass
class DerivedFieldDescriptor(FieldDescriptor):
    """A class to describe the origin of a field in a dataset as a derived field."""
    description: str
    constituent_fields: List[FieldDescriptor]

    def __str__(self):
        string = f"Derived by {self.description} from the following fields:\n"
        for descriptor in self.constituent_fields:
            string += f"\t{descriptor}\n"
        return string
