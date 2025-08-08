"""
Postprocessing module for WaterSoft Hydrological ML.

This module contains functions for calculating drought indices (SSI),
reservoir operations analysis, and adaptive management strategies.
"""

from .ssi_calculation import (
    calculate_ssi,
    calculate_standardized_index_monthly,
    get_ssi_category,
    plot_ssi_timeseries
)

from .reservoir_operations import (
    calculate_adaptive_releases,
    derive_empirical_factors,
    load_operational_parameters,
    run_adaptive_management
)

__all__ = [
    "calculate_ssi",
    "calculate_standardized_index_monthly", 
    "get_ssi_category",
    "plot_ssi_timeseries",
    "calculate_adaptive_releases",
    "derive_empirical_factors",
    "load_operational_parameters",
    "run_adaptive_management"
]
