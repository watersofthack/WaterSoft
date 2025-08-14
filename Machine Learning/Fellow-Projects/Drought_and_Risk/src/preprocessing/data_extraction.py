"""
Data extraction module for CAMELS dataset and USGS streamflow data.

This module provides functions to extract and download hydrological and
meteorological data from various sources including USGS and CAMELS dataset.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import pygeohydro as gh
from pygeohydro import NWIS
from pynhd import WaterData
import os
import shutil
import glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_camels_data(
    gauge_list: List[str],
    start_date: str = "1980-01-01",
    end_date: str = "2024-12-31",
    save_path: str = "data/raw/time_series"
) -> None:
    """
    Extract streamflow data from USGS for a list of gauge stations.
    
    Args:
        gauge_list: List of USGS gauge station IDs
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        save_path: Directory to save the extracted data
    """
    logger.info(f"Extracting streamflow data for {len(gauge_list)} stations")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Create save directory if it doesn't exist
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    nwis = NWIS()
    
    for i, gauge_id in enumerate(gauge_list):
        try:
            logger.info(f"Processing station {gauge_id} ({i+1}/{len(gauge_list)})")
            gauge_id_str = str(gauge_id)
            
            # Get streamflow data
            q_cms = nwis.get_streamflow(gauge_id_str, (start_date, end_date))
            
            if q_cms is not None and not q_cms.empty:
                # Save to CSV
                output_file = os.path.join(save_path, f"{gauge_id_str}.csv")
                q_cms.to_csv(output_file)
                logger.info(f"Saved {len(q_cms)} records to {output_file}")
            else:
                logger.warning(f"No data retrieved for station {gauge_id_str}")
                
        except Exception as e:
            logger.error(f"Error processing station {gauge_id}: {str(e)}")
            continue
    
    logger.info("Data extraction completed")


def get_streamflow_data(
    station_id: str,
    start_date: str,
    end_date: str,
    data_dir: str = "data/raw/time_series"
) -> pd.DataFrame:
    """
    Load streamflow data for a specific station.
    
    Args:
        station_id: USGS station ID
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_dir: Directory containing the data files
        
    Returns:
        DataFrame with streamflow data
    """
    file_path = os.path.join(data_dir, f"{station_id}.csv")
    
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Filter by date range
        df = df.loc[start_date:end_date]
        
        logger.info(f"Loaded {len(df)} records for station {station_id}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data for station {station_id}: {str(e)}")
        return pd.DataFrame()


def download_meteorological_data(
    gauge_list: List[str],
    start_date: str = "1980-01-01", 
    end_date: str = "2024-12-31",
    variables: List[str] = ["prcp", "tmax"],
    save_path: str = "data/raw/time_series"
) -> None:
    """
    Download meteorological data (precipitation, temperature) for gauge stations.
    
    Args:
        gauge_list: List of USGS gauge station IDs
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        variables: List of meteorological variables to download
        save_path: Directory to save the data
    """
    logger.info(f"Downloading meteorological data for {len(gauge_list)} stations")
    logger.info(f"Variables: {variables}")
    
    # Create save directory
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # This is a placeholder for meteorological data download
    # In practice, you would use services like Daymet, PRISM, or other APIs
    logger.warning("Meteorological data download not implemented")
    logger.info("Please use existing CAMELS dataset or implement specific data source")


def create_basin_list_from_shapefile(
    shapefile_path: str,
    id_column: str = "hru_id",
    output_file: str = "configs/basin_list.txt"
) -> List[str]:
    """
    Create a basin list from a shapefile containing catchment boundaries.
    
    Args:
        shapefile_path: Path to the shapefile
        id_column: Column name containing the basin IDs
        output_file: Output file to save the basin list
        
    Returns:
        List of basin IDs
    """
    try:
        # Read shapefile
        catchments = gpd.read_file(shapefile_path)
        gauge_list = catchments[id_column].astype(str).tolist()
        
        # Create output directory
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Save basin list
        with open(output_file, 'w') as f:
            for gauge_id in gauge_list:
                f.write(f"{gauge_id}\n")
        
        logger.info(f"Created basin list with {len(gauge_list)} stations")
        logger.info(f"Saved to: {output_file}")
        
        return gauge_list
        
    except Exception as e:
        logger.error(f"Error creating basin list: {str(e)}")
        return []


def validate_data_completeness(
    data_dir: str = "data/raw/time_series",
    start_date: str = "1980-01-01",
    end_date: str = "2023-12-31",
    min_completeness: float = 0.8
) -> Dict[str, float]:
    """
    Validate data completeness for all stations.
    
    Args:
        data_dir: Directory containing data files
        start_date: Start date for validation
        end_date: End date for validation
        min_completeness: Minimum required data completeness (0-1)
        
    Returns:
        Dictionary with station IDs and their completeness ratios
    """
    logger.info("Validating data completeness...")
    
    completeness_report = {}
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    # Calculate expected number of days
    expected_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    
    for file_path in csv_files:
        station_id = os.path.basename(file_path).replace('.csv', '')
        
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df = df.loc[start_date:end_date]
            
            # Calculate completeness
            actual_days = len(df.dropna())
            completeness = actual_days / expected_days
            completeness_report[station_id] = completeness
            
            if completeness < min_completeness:
                logger.warning(
                    f"Station {station_id}: {completeness:.2%} complete "
                    f"(below threshold of {min_completeness:.2%})"
                )
            else:
                logger.info(f"Station {station_id}: {completeness:.2%} complete")
                
        except Exception as e:
            logger.error(f"Error validating station {station_id}: {str(e)}")
            completeness_report[station_id] = 0.0
    
    # Summary statistics
    valid_stations = [k for k, v in completeness_report.items() if v >= min_completeness]
    logger.info(f"Data validation complete:")
    logger.info(f"  Total stations: {len(completeness_report)}")
    logger.info(f"  Valid stations: {len(valid_stations)}")
    logger.info(f"  Average completeness: {np.mean(list(completeness_report.values())):.2%}")
    
    return completeness_report


if __name__ == "__main__":
    # Example usage
    gauge_list = [
        "6221400", "6224000", "6614800", "6623800", "7083000",
        "9034900", "9035800", "9035900", "9047700", "9065500",
        "9066000", "9066200", "9066300", "9081600", "9107000",
        "9210500", "9223000", "9306242", "9312600", "9352900",
        "9378170", "9378630", "10205030", "13023000"
    ]
    
    # Extract streamflow data
    extract_camels_data(
        gauge_list=gauge_list,
        start_date="1980-01-01",
        end_date="2024-12-31",
        save_path="data/raw/time_series"
    )
    
    # Validate data completeness
    completeness = validate_data_completeness()
    
    print("Data extraction and validation completed!")
