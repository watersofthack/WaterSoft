#!/usr/bin/env python3
# coding: utf-8

"""
Retrieve NWM v3.0 retrospective streamflow for one ReachID and
convert it to estimated USGS-equivalent gauge height using a
station-specific USGS rating curve.

Default target:

    USGS station: 08210000
    NWM ReachID: 3168766

The NWM v3.0 retrospective archive covers:

    February 1979 through January 2023

Example: retrieve all of 2013

    python scripts/09_retrieve_nwm_retrospective_gauge_height.py \
        --site-id 08210000 \
        --reach-id 3168766 \
        --start 2013-01-01 \
        --end 2013-12-31

Example: retrieve one event

    python scripts/09_retrieve_nwm_retrospective_gauge_height.py \
        --site-id 08210000 \
        --reach-id 3168766 \
        --start 2013-05-20 \
        --end 2013-06-10

Requirements:

    python -m pip install -r requirements.txt

The project pins Zarr 2.x and NumPy 1.x because the current Xarray
workflow and the rest of the scientific environment use those APIs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import xarray as xr


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

CFS_PER_CMS = 35.3146667215
METERS_PER_FOOT = 0.3048

DEFAULT_SITE_ID = "08210000"
DEFAULT_REACH_ID = 3168766

NWM_RETROSPECTIVE_ZARR = (
    "s3://noaa-nwm-retrospective-3-0-pds/"
    "CONUS/zarr/chrtout.zarr"
)

NWM_RETROSPECTIVE_START = pd.Timestamp(
    "1979-02-01 00:00:00"
)

NWM_RETROSPECTIVE_END = pd.Timestamp(
    "2023-01-31 23:00:00"
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RATING_DIR = (
    ROOT
    / "data"
    / "processed"
    / "rating_curves"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "nwm"
    / "retrospective_gauge_height"
)


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Retrieve NWM v3.0 retrospective streamflow and "
            "convert it to USGS-equivalent gauge height."
        )
    )

    parser.add_argument(
        "--site-id",
        type=str,
        default=DEFAULT_SITE_ID,
        help=(
            "USGS site identifier. "
            f"Default: {DEFAULT_SITE_ID}"
        ),
    )

    parser.add_argument(
        "--reach-id",
        type=int,
        default=DEFAULT_REACH_ID,
        help=(
            "NWM feature_id / ReachID. "
            f"Default: {DEFAULT_REACH_ID}"
        ),
    )

    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help=(
            "Inclusive starting date or datetime, for example "
            "2013-01-01 or 2013-01-01T00:00."
        ),
    )

    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help=(
            "Inclusive ending date or datetime, for example "
            "2013-12-31 or 2013-12-31T23:00."
        ),
    )

    parser.add_argument(
        "--rating-file",
        type=Path,
        default=None,
        help=(
            "Processed USGS discharge-to-stage rating CSV. "
            "Default: data/processed/rating_curves/"
            "<site_id>_discharge_to_stage_rating.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output directory for the retrospective CSV."
        ),
    )

    parser.add_argument(
        "--allow-boundary-clipping",
        action="store_true",
        help=(
            "Assign rating-curve boundary stage values when "
            "discharge is outside the published rating range. "
            "By default, out-of-range stage is NaN."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# General validation
# ---------------------------------------------------------------------

def normalize_site_id(site_id: str) -> str:
    """Normalize a USGS site number."""

    cleaned = str(site_id).strip()

    if not cleaned.isdigit():
        raise ValueError(
            "USGS site ID must contain digits only: "
            f"{site_id!r}"
        )

    return cleaned.zfill(8)


def parse_time_range(
    start_text: str,
    end_text: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Parse and validate an inclusive retrospective time range.

    Date-only end values are interpreted as ending at 23:00 UTC.
    """

    start = pd.Timestamp(start_text)
    end = pd.Timestamp(end_text)

    if len(str(end_text).strip()) == 10:
        end = end + pd.Timedelta(hours=23)

    start = start.floor("h")
    end = end.floor("h")

    if start > end:
        raise ValueError(
            f"Start time {start} occurs after end time {end}."
        )

    if start < NWM_RETROSPECTIVE_START:
        raise ValueError(
            "Requested start predates the NWM v3.0 retrospective "
            f"archive: {NWM_RETROSPECTIVE_START}."
        )

    if end > NWM_RETROSPECTIVE_END:
        raise ValueError(
            "Requested end exceeds the NWM v3.0 retrospective "
            f"archive: {NWM_RETROSPECTIVE_END}."
        )

    return start, end


# ---------------------------------------------------------------------
# Rating curve
# ---------------------------------------------------------------------

def load_rating_curve(
    rating_file: Path,
    expected_site_id: str,
    expected_reach_id: int,
) -> pd.DataFrame:
    """Load and validate a discharge-to-stage rating table."""

    if not rating_file.exists():
        raise FileNotFoundError(
            f"Rating-curve file not found: {rating_file}"
        )

    rating = pd.read_csv(
        rating_file,
        dtype={"site_id": "string"},
    )

    required_columns = {
        "discharge_cfs",
        "gauge_height_ft",
    }

    missing = required_columns.difference(
        rating.columns
    )

    if missing:
        raise ValueError(
            "Rating table is missing required columns: "
            f"{sorted(missing)}"
        )

    rating["discharge_cfs"] = pd.to_numeric(
        rating["discharge_cfs"],
        errors="coerce",
    )

    rating["gauge_height_ft"] = pd.to_numeric(
        rating["gauge_height_ft"],
        errors="coerce",
    )

    rating = rating.dropna(
        subset=[
            "discharge_cfs",
            "gauge_height_ft",
        ]
    )

    rating = rating.loc[
        rating["discharge_cfs"] >= 0
    ].copy()

    rating = (
        rating.sort_values(
            [
                "discharge_cfs",
                "gauge_height_ft",
            ]
        )
        .drop_duplicates(
            subset=["discharge_cfs"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if len(rating) < 2:
        raise ValueError(
            "At least two valid rating points are required."
        )

    if "site_id" in rating.columns:
        rating["site_id"] = (
            rating["site_id"]
            .astype("string")
            .str.zfill(8)
        )

        rating_sites = set(
            rating["site_id"]
            .dropna()
            .unique()
            .tolist()
        )

        if (
            rating_sites
            and expected_site_id not in rating_sites
        ):
            raise ValueError(
                f"Rating file is not for USGS "
                f"{expected_site_id}. Found: "
                f"{sorted(rating_sites)}"
            )

    if "nwm_reach_id" in rating.columns:
        available_reaches = set(
            pd.to_numeric(
                rating["nwm_reach_id"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if (
            available_reaches
            and expected_reach_id not in available_reaches
        ):
            raise ValueError(
                f"Rating file is not associated with ReachID "
                f"{expected_reach_id}. Found: "
                f"{sorted(available_reaches)}"
            )

    return rating


def convert_discharge_to_gauge_height(
    retrospective: pd.DataFrame,
    rating: pd.DataFrame,
    allow_boundary_clipping: bool = False,
) -> pd.DataFrame:
    """Convert retrospective NWM streamflow to estimated stage."""

    result = retrospective.copy()

    result["streamflow_cms"] = pd.to_numeric(
        result["streamflow_cms"],
        errors="coerce",
    )

    result["streamflow_cfs"] = (
        result["streamflow_cms"]
        * CFS_PER_CMS
    )

    minimum_flow = float(
        rating["discharge_cfs"].min()
    )

    maximum_flow = float(
        rating["discharge_cfs"].max()
    )

    minimum_stage = float(
        rating.iloc[0]["gauge_height_ft"]
    )

    maximum_stage = float(
        rating.iloc[-1]["gauge_height_ft"]
    )

    if allow_boundary_clipping:
        left_stage = minimum_stage
        right_stage = maximum_stage
    else:
        left_stage = np.nan
        right_stage = np.nan

    result["estimated_gauge_height_ft"] = np.interp(
        result["streamflow_cfs"],
        rating["discharge_cfs"],
        rating["gauge_height_ft"],
        left=left_stage,
        right=right_stage,
    )

    result["estimated_gauge_height_m"] = (
        result["estimated_gauge_height_ft"]
        * METERS_PER_FOOT
    )

    result["within_usgs_rating_range"] = (
        result["streamflow_cfs"].between(
            minimum_flow,
            maximum_flow,
            inclusive="both",
        )
    )

    result["below_usgs_rating_range"] = (
        result["streamflow_cfs"]
        < minimum_flow
    )

    result["above_usgs_rating_range"] = (
        result["streamflow_cfs"]
        > maximum_flow
    )

    result["rating_minimum_discharge_cfs"] = (
        minimum_flow
    )

    result["rating_maximum_discharge_cfs"] = (
        maximum_flow
    )

    result["rating_minimum_gauge_height_ft"] = (
        minimum_stage
    )

    result["rating_maximum_gauge_height_ft"] = (
        maximum_stage
    )

    return result


# ---------------------------------------------------------------------
# NWM retrospective retrieval
# ---------------------------------------------------------------------

def open_nwm_retrospective() -> tuple[
    xr.Dataset,
    s3fs.S3FileSystem,
]:
    """
    Open the NWM v3.0 CHRTOUT Zarr archive lazily.

    Returns
    -------
    dataset
        Open remote NWM retrospective dataset.
    filesystem
        S3 filesystem that must be closed after retrieval.
    """

    filesystem = s3fs.S3FileSystem(
        anon=True,
        default_fill_cache=False,
        default_cache_type="none",
        skip_instance_cache=True,
    )

    store = s3fs.S3Map(
        root=NWM_RETROSPECTIVE_ZARR,
        s3=filesystem,
        check=False,
    )

    print(
        "Opening NWM v3.0 retrospective Zarr archive..."
    )

    dataset = xr.open_zarr(
        store,
        consolidated=True,
        chunks={},
    )

    return dataset, filesystem


def identify_time_coordinate(
    dataset: xr.Dataset,
) -> str:
    """Identify the temporal coordinate in the Zarr dataset."""

    candidates = [
        "time",
        "reference_time",
        "valid_time",
    ]

    for candidate in candidates:
        if (
            candidate in dataset.coords
            or candidate in dataset.dims
        ):
            return candidate

    raise KeyError(
        "Could not identify the NWM time coordinate. "
        f"Coordinates: {list(dataset.coords)}; "
        f"dimensions: {list(dataset.dims)}"
    )


def identify_feature_coordinate(
    dataset: xr.Dataset,
) -> str:
    """Identify the NWM reach coordinate."""

    candidates = [
        "feature_id",
        "feature_ids",
    ]

    for candidate in candidates:
        if (
            candidate in dataset.coords
            or candidate in dataset.variables
        ):
            return candidate

    raise KeyError(
        "Could not identify the NWM feature coordinate. "
        f"Variables include: {list(dataset.variables)[:30]}"
    )


def select_reach(
    dataset: xr.Dataset,
    reach_id: int,
    feature_coordinate: str,
) -> xr.Dataset:
    """
    Select one NWM reach.

    First try coordinate-based selection. If feature_id is stored
    as a data variable rather than an indexed coordinate, find its
    positional index.
    """

    if feature_coordinate in dataset.indexes:
        try:
            return dataset.sel(
                {
                    feature_coordinate: reach_id
                }
            )
        except KeyError as exc:
            raise KeyError(
                f"ReachID {reach_id} was not found in the "
                "NWM retrospective dataset."
            ) from exc

    feature_values = dataset[
        feature_coordinate
    ].values

    matching_indices = np.flatnonzero(
        feature_values == reach_id
    )

    if len(matching_indices) == 0:
        raise KeyError(
            f"ReachID {reach_id} was not found in the "
            "NWM retrospective dataset."
        )

    feature_variable = dataset[
        feature_coordinate
    ]

    if feature_variable.ndim != 1:
        raise ValueError(
            f"Unexpected {feature_coordinate} dimensions: "
            f"{feature_variable.dims}"
        )

    reach_dimension = feature_variable.dims[0]

    return dataset.isel(
        {
            reach_dimension: int(
                matching_indices[0]
            )
        }
    )


def retrieve_nwm_retrospective(
    reach_id: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Retrieve hourly NWM retrospective streamflow."""

    dataset, filesystem = open_nwm_retrospective()

    try:
        if "streamflow" not in dataset:
            raise KeyError(
                "The Zarr dataset does not contain streamflow. "
                f"Available variables include: "
                f"{list(dataset.data_vars)[:30]}"
            )

        time_coordinate = identify_time_coordinate(
            dataset
        )

        feature_coordinate = identify_feature_coordinate(
            dataset
        )

        print(
            f"Time coordinate:    {time_coordinate}"
        )

        print(
            f"Feature coordinate: {feature_coordinate}"
        )

        reach_data = select_reach(
            dataset=dataset,
            reach_id=reach_id,
            feature_coordinate=feature_coordinate,
        )

        reach_data = reach_data[
            ["streamflow"]
        ].sel(
            {
                time_coordinate: slice(
                    start,
                    end,
                )
            }
        )

        print(
            f"Retrieving ReachID {reach_id} from "
            f"{start} through {end}..."
        )

        reach_data = reach_data.compute()

        frame = (
            reach_data["streamflow"]
            .to_dataframe(
                name="streamflow_cms"
            )
            .reset_index()
        )

        if time_coordinate != "datetime":
            frame = frame.rename(
                columns={
                    time_coordinate: "datetime"
                }
            )

        frame["datetime"] = pd.to_datetime(
            frame["datetime"],
            errors="coerce",
            utc=True,
        )

        frame["feature_id"] = reach_id

        frame = frame[
            [
                "datetime",
                "feature_id",
                "streamflow_cms",
            ]
        ]

        frame = frame.dropna(
            subset=[
                "datetime",
                "streamflow_cms",
            ]
        )
        frame = frame.sort_values(
            "datetime"
        ).reset_index(drop=True)
        return frame

    finally:
        # S3FS manages its asynchronous client with a finalizer.
        # Private connection attributes differ between S3FS versions,
        # so cleanup must not depend on `_s3creator` being present or
        # mask an otherwise successful retrieval.
        try:
            dataset.close()
        except Exception as exc:
            print(
                "Warning: could not close the NWM dataset "
                f"cleanly: {exc}"
            )

        try:
            filesystem.clear_instance_cache()
        except Exception as exc:
            print(
                "Warning: could not clear the S3FS instance "
                f"cache: {exc}"
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    site_id = normalize_site_id(
        args.site_id
    )

    start, end = parse_time_range(
        start_text=args.start,
        end_text=args.end,
    )

    if args.rating_file is None:
        rating_file = (
            DEFAULT_RATING_DIR
            / (
                f"{site_id}_"
                "discharge_to_stage_rating.csv"
            )
        )
    else:
        rating_file = args.rating_file.resolve()

    output_dir = args.output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"USGS site:       {site_id}"
    )

    print(
        f"NWM ReachID:     {args.reach_id}"
    )

    print(
        f"Starting time:   {start}"
    )

    print(
        f"Ending time:     {end}"
    )

    print(
        f"Rating file:     {rating_file}"
    )

    rating = load_rating_curve(
        rating_file=rating_file,
        expected_site_id=site_id,
        expected_reach_id=args.reach_id,
    )

    print(
        "\nRating curve:"
    )

    print(
        f"  Points:              {len(rating):,}"
    )

    print(
        "  Discharge range:     "
        f"{rating['discharge_cfs'].min():,.3f} to "
        f"{rating['discharge_cfs'].max():,.3f} ft³/s"
    )

    print(
        "  Gauge-height range:  "
        f"{rating['gauge_height_ft'].min():,.3f} to "
        f"{rating['gauge_height_ft'].max():,.3f} ft"
    )

    retrospective = retrieve_nwm_retrospective(
        reach_id=args.reach_id,
        start=start,
        end=end,
    )

    if retrospective.empty:
        raise RuntimeError(
            "The NWM retrospective query returned no data."
        )

    result = convert_discharge_to_gauge_height(
        retrospective=retrospective,
        rating=rating,
        allow_boundary_clipping=(
            args.allow_boundary_clipping
        ),
    )

    result.insert(
        0,
        "site_id",
        site_id,
    )

    result["nwm_version"] = (
        "3.0_retrospective"
    )

    result["rating_curve_site_id"] = (
        site_id
    )

    result["rating_curve_method"] = (
        "linear_interpolation_discharge_to_stage"
    )

    start_tag = start.strftime(
        "%Y%m%d%H"
    )

    end_tag = end.strftime(
        "%Y%m%d%H"
    )

    output_path = (
        output_dir
        / (
            f"USGS_{site_id}_"
            f"ReachID_{args.reach_id}_"
            f"NWM_v3_retrospective_"
            f"{start_tag}_{end_tag}_"
            "streamflow_gauge_height.csv"
        )
    )

    result.to_csv(
        output_path,
        index=False,
    )

    expected_hours = int(
        (
            end - start
        ).total_seconds()
        / 3600
    ) + 1

    available_stage_count = int(
        result[
            "estimated_gauge_height_ft"
        ].notna().sum()
    )

    outside_rating_count = int(
        (
            ~result[
                "within_usgs_rating_range"
            ]
        ).sum()
    )

    print(
        "\nRetrieval summary:"
    )

    print(
        f"  Expected hourly rows:       {expected_hours:,}"
    )

    print(
        f"  Retrieved rows:             {len(result):,}"
    )

    print(
        f"  Gauge heights available:    {available_stage_count:,}"
    )

    print(
        f"  Outside rating range:       {outside_rating_count:,}"
    )

    print(
        "  Minimum streamflow:         "
        f"{result['streamflow_cfs'].min():,.3f} ft³/s"
    )

    print(
        "  Maximum streamflow:         "
        f"{result['streamflow_cfs'].max():,.3f} ft³/s"
    )

    print(
        "  Minimum estimated stage:    "
        f"{result['estimated_gauge_height_ft'].min():,.3f} ft"
    )

    print(
        "  Maximum estimated stage:    "
        f"{result['estimated_gauge_height_ft'].max():,.3f} ft"
    )

    print(
        f"\nSaved: {output_path}"
    )

if __name__ == "__main__":
    main()
