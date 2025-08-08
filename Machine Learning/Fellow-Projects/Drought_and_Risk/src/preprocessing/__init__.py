"""
Data preprocessing module for WaterSoft Hydrological ML.

This module contains functions for extracting, processing, and preparing
hydrological and meteorological data for machine learning models.
"""

from .data_extraction import (
    extract_camels_data,
    get_streamflow_data,
    download_meteorological_data
)

from .attribute_processing import (
    process_basin_attributes,
    normalize_attributes,
    create_attribute_embeddings
)

__all__ = [
    "extract_camels_data",
    "get_streamflow_data", 
    "download_meteorological_data",
    "process_basin_attributes",
    "normalize_attributes",
    "create_attribute_embeddings"
]
