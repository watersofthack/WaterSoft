"""
WaterSoft Hydrological ML Package

A comprehensive machine learning framework for hydrological modeling,
combining LSTM and Transformer architectures for streamflow prediction
with adaptive reservoir management strategies.
"""

__version__ = "1.0.0"
__author__ = "WaterSoft Hydrological ML Team"
__email__ = "contact@watersoft-ml.org"

from . import preprocessing
from . import models
from . import postprocessing
from . import utils

__all__ = [
    "preprocessing",
    "models", 
    "postprocessing",
    "utils"
]
