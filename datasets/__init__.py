"""Dataset-independent loading utilities for PCRFusion."""

from .registry import load_dataset
from .spec import DatasetSpec

__all__ = ["load_dataset", "DatasetSpec"]
