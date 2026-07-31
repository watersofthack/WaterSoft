#!/usr/bin/env python3
# coding: utf-8

"""
Retrieve a rolling operational NWM streamflow series and convert it to
USGS-equivalent gauge height.

The output combines:

1. hourly Analysis and Assimilation (analysis_assim) values for the
   previous seven days, and
2. the latest complete Short Range forecast for the next 18 hours.

The two products are joined at the Short Range initialization time.
Analysis supplies the initialization hour; Short Range starts at lead
hour 1, so the combined time series has no duplicate boundary row.

Default target:

    USGS station: 08210000
    NWM ReachID: 3168766

Example:

    python scripts/RetrieveNWM.py

Reproducible historical invocation:

    python scripts/RetrieveNWM.py --as-of 2026-07-30T20:00:00Z

Requirements:

    python -m pip install pandas numpy xarray s3fs h5netcdf dask h5py
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import xarray as xr


CFS_PER_CMS = 35.3146667215
METERS_PER_FOOT = 0.3048

DEFAULT_SITE_ID = "08210000"
DEFAULT_REACH_ID = 3168766
DEFAULT_HISTORY_DAYS = 7
DEFAULT_FORECAST_HOURS = 18

NWM_BUCKET = "noaa-nwm-pds"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATING_DIR = ROOT / "data" / "processed" / "rating_curves"
DEFAULT_OUTPUT_DIR = (
    ROOT / "data" / "processed" / "nwm" / "operational_gauge_height"
)

SHORT_RANGE_PATTERN = re.compile(
    r"nwm\.(?P<date>\d{8})/short_range/"
    r"nwm\.t(?P<cycle>\d{2})z\.short_range\.channel_rt\."
    r"f(?P<lead>\d{3})\.conus\.nc$"
)


def read_nwm_streamflow_file(
    request: tuple[str, int, int],
) -> dict[str, object]:
    """Read one reach by positional index from one NWM channel file."""

    path, feature_index, reach_id = request
    filesystem = s3fs.S3FileSystem(
        anon=True,
        default_fill_cache=False,
        default_cache_type="none",
        skip_instance_cache=True,
    )

    with filesystem.open(path, mode="rb") as remote_file:
        with xr.open_dataset(remote_file, engine="h5netcdf") as dataset:
            if "streamflow" not in dataset:
                raise KeyError(
                    f"NWM channel file does not contain streamflow: {path}"
                )

            streamflow_variable = dataset["streamflow"]
            if "feature_id" not in streamflow_variable.dims:
                raise ValueError(
                    "Unexpected streamflow dimensions in "
                    f"{path}: {streamflow_variable.dims}"
                )

            streamflow = float(
                streamflow_variable.isel(feature_id=feature_index).load().item()
            )
            valid_time = dataset.attrs.get("model_output_valid_time")
            if valid_time is None:
                raise KeyError(
                    "NWM file is missing model_output_valid_time: "
                    f"{path}"
                )

    return {
        "datetime": valid_time,
        "feature_id": reach_id,
        "streamflow_cms": streamflow,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve recent NWM analysis and the latest short-range "
            "forecast, then convert streamflow to gauge height."
        )
    )
    parser.add_argument(
        "--site-id",
        default=DEFAULT_SITE_ID,
        help=f"USGS site ID. Default: {DEFAULT_SITE_ID}",
    )
    parser.add_argument(
        "--reach-id",
        type=int,
        default=DEFAULT_REACH_ID,
        help=f"NWM feature_id. Default: {DEFAULT_REACH_ID}",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help=f"Previous analysis days. Default: {DEFAULT_HISTORY_DAYS}",
    )
    parser.add_argument(
        "--forecast-hours",
        type=int,
        default=DEFAULT_FORECAST_HOURS,
        help=(
            "Short-range lead hours to retrieve. "
            f"Default: {DEFAULT_FORECAST_HOURS}"
        ),
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help=(
            "Latest allowable cycle time in UTC. Default: current UTC "
            "time. Example: 2026-07-30T20:00:00Z"
        ),
    )
    parser.add_argument(
        "--rating-file",
        type=Path,
        default=None,
        help=(
            "USGS discharge-to-stage rating CSV. Default: "
            "data/processed/rating_curves/"
            "<site_id>_discharge_to_stage_rating.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the combined output CSV.",
    )
    parser.add_argument(
        "--allow-boundary-clipping",
        action="store_true",
        help=(
            "Use the nearest rating-curve boundary stage for flows "
            "outside the published range. By default, stage is NaN."
        ),
    )
    return parser.parse_args()


def normalize_site_id(site_id: str) -> str:
    cleaned = str(site_id).strip()
    if not cleaned.isdigit():
        raise ValueError(f"USGS site ID must contain digits only: {site_id!r}")
    return cleaned.zfill(8)


def parse_as_of(value: str | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC").floor("h")

    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed.floor("h")


def load_rating_curve(
    rating_file: Path,
    expected_site_id: str,
    expected_reach_id: int,
) -> pd.DataFrame:
    """Load and validate the station-specific USGS rating curve."""

    if not rating_file.exists():
        raise FileNotFoundError(f"Rating-curve file not found: {rating_file}")

    rating = pd.read_csv(rating_file, dtype={"site_id": "string"})
    required = {"discharge_cfs", "gauge_height_ft"}
    missing = required.difference(rating.columns)
    if missing:
        raise ValueError(
            f"Rating table is missing required columns: {sorted(missing)}"
        )

    for column in required:
        rating[column] = pd.to_numeric(rating[column], errors="coerce")

    rating = (
        rating.dropna(subset=list(required))
        .loc[lambda frame: frame["discharge_cfs"] >= 0]
        .sort_values(["discharge_cfs", "gauge_height_ft"])
        .drop_duplicates("discharge_cfs", keep="last")
        .reset_index(drop=True)
    )
    if len(rating) < 2:
        raise ValueError("At least two valid rating points are required.")

    if "site_id" in rating:
        sites = set(
            rating["site_id"].astype("string").str.zfill(8).dropna().unique()
        )
        if sites and expected_site_id not in sites:
            raise ValueError(
                f"Rating file is not for USGS {expected_site_id}. "
                f"Found: {sorted(sites)}"
            )

    if "nwm_reach_id" in rating:
        reaches = set(
            pd.to_numeric(rating["nwm_reach_id"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        if reaches and expected_reach_id not in reaches:
            raise ValueError(
                f"Rating file is not for ReachID {expected_reach_id}. "
                f"Found: {sorted(reaches)}"
            )

    return rating


def convert_discharge_to_gauge_height(
    data: pd.DataFrame,
    rating: pd.DataFrame,
    allow_boundary_clipping: bool,
) -> pd.DataFrame:
    """Apply linear interpolation to the USGS discharge-stage rating."""

    result = data.copy()
    result["streamflow_cms"] = pd.to_numeric(
        result["streamflow_cms"], errors="coerce"
    )
    result["streamflow_cfs"] = result["streamflow_cms"] * CFS_PER_CMS

    minimum_flow = float(rating["discharge_cfs"].min())
    maximum_flow = float(rating["discharge_cfs"].max())
    minimum_stage = float(rating.iloc[0]["gauge_height_ft"])
    maximum_stage = float(rating.iloc[-1]["gauge_height_ft"])

    left_stage = minimum_stage if allow_boundary_clipping else np.nan
    right_stage = maximum_stage if allow_boundary_clipping else np.nan

    result["estimated_gauge_height_ft"] = np.interp(
        result["streamflow_cfs"],
        rating["discharge_cfs"],
        rating["gauge_height_ft"],
        left=left_stage,
        right=right_stage,
    )
    result["estimated_gauge_height_m"] = (
        result["estimated_gauge_height_ft"] * METERS_PER_FOOT
    )
    result["within_usgs_rating_range"] = result["streamflow_cfs"].between(
        minimum_flow, maximum_flow, inclusive="both"
    )
    result["below_usgs_rating_range"] = (
        result["streamflow_cfs"] < minimum_flow
    )
    result["above_usgs_rating_range"] = (
        result["streamflow_cfs"] > maximum_flow
    )
    result["rating_minimum_discharge_cfs"] = minimum_flow
    result["rating_maximum_discharge_cfs"] = maximum_flow
    result["rating_minimum_gauge_height_ft"] = minimum_stage
    result["rating_maximum_gauge_height_ft"] = maximum_stage
    return result


def find_latest_complete_short_range_cycle(
    filesystem: s3fs.S3FileSystem,
    as_of: pd.Timestamp,
    forecast_hours: int,
    lookback_days: int = 3,
) -> tuple[pd.Timestamp, list[str]]:
    """Find the newest cycle containing every requested forecast lead."""

    cycle_files: dict[pd.Timestamp, dict[int, str]] = {}

    for offset in range(lookback_days + 1):
        date = (as_of - pd.Timedelta(days=offset)).strftime("%Y%m%d")
        pattern = (
            f"{NWM_BUCKET}/nwm.{date}/short_range/"
            "nwm.t??z.short_range.channel_rt.f???.conus.nc"
        )

        for path in filesystem.glob(pattern):
            match = SHORT_RANGE_PATTERN.search(path)
            if not match:
                continue

            cycle = pd.Timestamp(
                f"{match.group('date')} {match.group('cycle')}:00:00",
                tz="UTC",
            )
            lead = int(match.group("lead"))
            if cycle <= as_of:
                cycle_files.setdefault(cycle, {})[lead] = path

    required_leads = set(range(1, forecast_hours + 1))
    complete_cycles = [
        cycle
        for cycle, files in cycle_files.items()
        if required_leads.issubset(files)
    ]
    if not complete_cycles:
        raise FileNotFoundError(
            f"No complete NWM Short Range cycle with leads 1 through "
            f"{forecast_hours} was found at or before {as_of}."
        )

    latest_cycle = max(complete_cycles)
    paths = [
        cycle_files[latest_cycle][lead]
        for lead in range(1, forecast_hours + 1)
    ]
    return latest_cycle, paths


def build_analysis_paths(
    filesystem: s3fs.S3FileSystem,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[str]:
    """Build and validate hourly analysis_assim tm00 file paths."""

    paths: list[str] = []
    missing: list[str] = []

    for cycle in pd.date_range(start, end, freq="1h", tz="UTC"):
        date = cycle.strftime("%Y%m%d")
        hour = cycle.strftime("%H")
        path = (
            f"{NWM_BUCKET}/nwm.{date}/analysis_assim/"
            f"nwm.t{hour}z.analysis_assim.channel_rt.tm00.conus.nc"
        )
        if filesystem.exists(path):
            paths.append(path)
        else:
            missing.append(path)

    if missing:
        example = "\n".join(f"  {path}" for path in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} hourly analysis files are missing. "
            f"First missing paths:\n{example}"
        )
    return paths


def retrieve_streamflow_files(
    paths: list[str],
    reach_id: int,
) -> pd.DataFrame:
    """Read streamflow for one reach from multiple NWM channel files."""

    filesystem = s3fs.S3FileSystem(
        anon=True,
        default_fill_cache=False,
        default_cache_type="none",
        skip_instance_cache=True,
    )

    # Resolve the ReachID position once. Repeating label-based `.sel()`
    # would re-read the roughly 2.7-million-element feature coordinate
    # from every operational file.
    with filesystem.open(paths[0], mode="rb") as remote_file:
        with xr.open_dataset(remote_file, engine="h5netcdf") as dataset:
            if "feature_id" not in dataset:
                raise KeyError("NWM channel file has no feature_id coordinate.")
            feature_values = dataset["feature_id"].values
            matches = np.flatnonzero(feature_values == reach_id)
            if len(matches) == 0:
                raise KeyError(f"ReachID {reach_id} was not found in NWM.")
            feature_index = int(matches[0])

    requests = [
        (path, feature_index, reach_id)
        for path in paths
    ]
    worker_count = min(8, len(requests))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        rows = list(executor.map(read_nwm_streamflow_file, requests))

    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(
        frame["datetime"],
        format="%Y-%m-%d_%H:%M:%S",
        errors="coerce",
        utc=True,
    )
    return (
        frame[["datetime", "feature_id", "streamflow_cms"]]
        .dropna(subset=["datetime", "streamflow_cms"])
        .drop_duplicates("datetime", keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()

    if args.history_days < 1:
        raise ValueError("--history-days must be at least 1.")
    if not 1 <= args.forecast_hours <= 18:
        raise ValueError("--forecast-hours must be between 1 and 18.")

    site_id = normalize_site_id(args.site_id)
    as_of = parse_as_of(args.as_of)
    rating_file = (
        args.rating_file.resolve()
        if args.rating_file
        else DEFAULT_RATING_DIR
        / f"{site_id}_discharge_to_stage_rating.csv"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"USGS site:         {site_id}")
    print(f"NWM ReachID:       {args.reach_id}")
    print(f"Latest cycle time: {as_of}")
    print(f"Rating curve:      {rating_file}")

    rating = load_rating_curve(
        rating_file=rating_file,
        expected_site_id=site_id,
        expected_reach_id=args.reach_id,
    )

    filesystem = s3fs.S3FileSystem(
        anon=True,
        default_fill_cache=False,
        default_cache_type="none",
    )

    forecast_init, forecast_paths = (
        find_latest_complete_short_range_cycle(
            filesystem=filesystem,
            as_of=as_of,
            forecast_hours=args.forecast_hours,
        )
    )
    analysis_start = forecast_init - pd.Timedelta(days=args.history_days)
    analysis_paths = build_analysis_paths(
        filesystem=filesystem,
        start=analysis_start,
        end=forecast_init,
    )

    print(f"\nSelected Short Range cycle: {forecast_init}")
    print(
        f"Analysis period:            {analysis_start} through "
        f"{forecast_init}"
    )
    print(
        f"Forecast period:            "
        f"{forecast_init + pd.Timedelta(hours=1)} through "
        f"{forecast_init + pd.Timedelta(hours=args.forecast_hours)}"
    )
    print(f"Analysis files:             {len(analysis_paths):,}")
    print(f"Forecast files:             {len(forecast_paths):,}")

    print("\nRetrieving Analysis and Assimilation streamflow...")
    analysis = retrieve_streamflow_files(analysis_paths, args.reach_id)
    analysis["data_type"] = "analysis_assim"
    analysis["init_time"] = analysis["datetime"]
    analysis["lead_time_hours"] = 0

    print("Retrieving Short Range streamflow...")
    forecast = retrieve_streamflow_files(forecast_paths, args.reach_id)
    forecast["data_type"] = "short_range"
    forecast["init_time"] = forecast_init
    forecast["lead_time_hours"] = (
        (forecast["datetime"] - forecast_init)
        .dt.total_seconds()
        .div(3600)
        .astype(int)
    )

    combined = pd.concat([analysis, forecast], ignore_index=True)
    combined = (
        combined.sort_values(["datetime", "data_type"])
        .drop_duplicates("datetime", keep="last")
        .reset_index(drop=True)
    )

    result = convert_discharge_to_gauge_height(
        data=combined,
        rating=rating,
        allow_boundary_clipping=args.allow_boundary_clipping,
    )
    result.insert(0, "site_id", site_id)
    result["rating_curve_site_id"] = site_id
    result["rating_curve_method"] = (
        "linear_interpolation_discharge_to_stage"
    )

    column_order = [
        "site_id",
        "datetime",
        "data_type",
        "init_time",
        "lead_time_hours",
        "feature_id",
        "streamflow_cms",
        "streamflow_cfs",
        "estimated_gauge_height_ft",
        "estimated_gauge_height_m",
        "within_usgs_rating_range",
        "below_usgs_rating_range",
        "above_usgs_rating_range",
        "rating_minimum_discharge_cfs",
        "rating_maximum_discharge_cfs",
        "rating_minimum_gauge_height_ft",
        "rating_maximum_gauge_height_ft",
        "rating_curve_site_id",
        "rating_curve_method",
    ]
    result = result[column_order]

    cycle_tag = forecast_init.strftime("%Y%m%d%H")
    output_path = (
        output_dir
        / (
            f"USGS_{site_id}_ReachID_{args.reach_id}_"
            f"NWM_analysis_{args.history_days}d_"
            f"short_range_{args.forecast_hours}h_{cycle_tag}_"
            "streamflow_gauge_height.csv"
        )
    )
    result.to_csv(output_path, index=False)

    expected_rows = args.history_days * 24 + 1 + args.forecast_hours
    print("\nRetrieval summary:")
    print(f"  Expected rows:           {expected_rows:,}")
    print(f"  Retrieved rows:          {len(result):,}")
    print(
        f"  Gauge heights available: "
        f"{result['estimated_gauge_height_ft'].notna().sum():,}"
    )
    print(
        f"  Outside rating range:    "
        f"{(~result['within_usgs_rating_range']).sum():,}"
    )
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
