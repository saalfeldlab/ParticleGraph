import abc
from dataclasses import dataclass
from typing import Dict

from astropy.units import Unit
import numpy as np


class FieldDescriptor(abc.ABC):
    """A class to describe the origin of a field in a dataset."""


@dataclass
class CsvDescriptor(FieldDescriptor):
    """A class to describe the origin of a field in a dataset as a column of a CSV file."""
    filename: str
    column_name: str
    type: np.dtype
    unit: Unit


@dataclass
class DerivedFieldDescriptor(FieldDescriptor):
    """A class to describe the origin of a field in a dataset as a derived field."""
    constituent_fields: Dict[str, FieldDescriptor]
    description: str
