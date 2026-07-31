#!/usr/bin/env python3
# coding: utf-8

"""
Analyze flood-wave propagation among upstream and tributary gauges
contributing to USGS 08210000.

The script identifies major-stage exceedance events at the target gauge,
ranks them by duration from longest to shortest, and plots streamflow and
gage-height time series for a selected event.

Examples
--------
Select the longest major event:

    python scripts/10_flood_wave_propagation.py \
        --event-rank 1

Select the second-longest event:

    python scripts/10_flood_wave_propagation.py \
        --event-rank 2

Use a wider time window:

    python scripts/10_flood_wave_propagation.py \
        --event-rank 1 \
        --days-before 5 \
        --days-after 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "raw"
    / "usgs"
    / "timeseries"
    / "usgs_gauge_height_streamflow_long.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "flood_wave_propagation"

FIGURE_DIR = ROOT / "docs" / "figures" / "flood_wave_propagation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


TARGET_SITE_ID = "08210000"

NWM_INPUT_DIR = ROOT / "data" / "processed" / "nwm" / "retrospective_gauge_height"

TARGET_REACH_ID = 3168766

LSTM_FORECAST_FILE = (
    ROOT / "data" / "processed" / "major_event_forecast_results.csv"
)

# Selected upstream main-stem and tributary gauges that are
# hydrologically relevant to the target gauge during the analyzed events.
# The target gauge is placed last in the subplot order.
NETWORK_SITE_IDS = [
    "08194000",
    "08194500",
    TARGET_SITE_ID,
]

EXPECTED_INTERVAL_MINUTES = 15
MAX_GAP_MINUTES = 30

THRESHOLD_COLUMNS = {
    "action": "action_stage_ft",
    "minor": "minor_flood_stage_ft",
    "moderate": "moderate_flood_stage_ft",
    "major": "major_flood_stage_ft",
}

THRESHOLD_COLORS = {
    "action": "#D6D600",  # yellow
    "minor": "#FFA500",  # orange
    "moderate": "#FF0000",  # red
    "major": "#C000C0",  # purple/magenta
}

THRESHOLD_LINE_STYLES = {
    "action": "-",
    "minor": "-",
    "moderate": "-",
    "major": "-",
}

HYDROGRAPH_LINE_WIDTH = 2
THRESHOLD_LINE_WIDTH = 0.75

# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze propagation of a selected major flood event "
            "toward USGS 08210000."
        )
    )

    parser.add_argument(
        "--event-rank",
        type=int,
        default=1,
        help=(
            "Major-event rank by duration at the target gauge. "
            "Rank 1 is the longest event, rank 2 is the "
            "second-longest, and so on. Default: 1"
        ),
    )

    parser.add_argument(
        "--days-before",
        type=float,
        default=3.0,
        help=(
            "Number of days before the selected target-gauge event "
            "to include. Default: 3"
        ),
    )

    parser.add_argument(
        "--days-after",
        type=float,
        default=3.0,
        help=(
            "Number of days after the selected target-gauge event "
            "to include. Default: 3"
        ),
    )

    parser.add_argument(
        "--resample",
        choices=["15min", "hourly"],
        default="hourly",
        help=(
            "Plotting resolution. Use '15min' for the original "
            "observations or 'hourly' for hourly means. Default: hourly"
        ),
    )

    parser.add_argument(
        "--nwm-file",
        type=Path,
        default=None,
        help=(
            "NWM retrospective streamflow/gauge-height CSV created "
            "by 09_retrieve_nwm_retrospective_gauge_height.py. "
            "When omitted, the script searches the default NWM "
            "retrospective output directory for a file containing "
            "the selected event window."
        ),
    )

    parser.add_argument(
        "--lstm-forecast-file",
        type=Path,
        default=LSTM_FORECAST_FILE,
        help=(
            "CSV containing LSTM stage forecasts by event_rank, "
            "issue_time, and forecast_time. Default: "
            "data/processed/major_event_forecast_results.csv"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def load_network_data(path: Path) -> pd.DataFrame:
    """Load streamflow, stage, metadata, and flood thresholds."""

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    required_columns = [
        "site_id",
        "site_name",
        "river",
        "datetime",
        "parameter_name",
        "value",
        *THRESHOLD_COLUMNS.values(),
    ]

    df = pd.read_csv(
        path,
        usecols=required_columns,
        dtype={"site_id": "string"},
        low_memory=False,
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
        utc=True,
    )

    df["site_id"] = df["site_id"].str.zfill(8)
    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    for column in THRESHOLD_COLUMNS.values():
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    available_sites = set(df["site_id"].dropna().unique())

    missing_sites = [
        site_id for site_id in NETWORK_SITE_IDS if site_id not in available_sites
    ]

    if missing_sites:
        raise ValueError(
            "The following requested gauges were not found: " + ", ".join(missing_sites)
        )

    df = df.loc[df["site_id"].isin(NETWORK_SITE_IDS)].copy()

    df = (
        df.drop_duplicates(
            subset=[
                "site_id",
                "datetime",
                "parameter_name",
            ],
            keep="last",
        )
        .sort_values(
            [
                "site_id",
                "datetime",
                "parameter_name",
            ]
        )
        .reset_index(drop=True)
    )

    return df


def load_lstm_event_forecasts(
    path: Path,
    event_rank: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """Load LSTM forecast trajectories for the selected ranked event."""

    if not path.exists():
        raise FileNotFoundError(f"LSTM forecast file not found: {path}")

    required_columns = [
        "event_rank",
        "issue_time",
        "lead_hour",
        "forecast_time",
        "predicted_stage_ft",
    ]

    forecasts = pd.read_csv(
        path,
        usecols=required_columns,
        low_memory=False,
    )

    forecasts["event_rank"] = pd.to_numeric(
        forecasts["event_rank"],
        errors="coerce",
    )
    forecasts["lead_hour"] = pd.to_numeric(
        forecasts["lead_hour"],
        errors="coerce",
    )
    forecasts["predicted_stage_ft"] = pd.to_numeric(
        forecasts["predicted_stage_ft"],
        errors="coerce",
    )

    for column in ["issue_time", "forecast_time"]:
        forecasts[column] = pd.to_datetime(
            forecasts[column],
            errors="coerce",
            utc=True,
        )

    available_ranks = sorted(
        forecasts["event_rank"].dropna().astype(int).unique().tolist()
    )
    if event_rank not in available_ranks:
        raise ValueError(
            f"LSTM forecast file does not contain event rank {event_rank}. "
            f"Available ranks: {available_ranks}"
        )

    event_forecasts = forecasts.loc[
        forecasts["event_rank"].eq(event_rank)
        & forecasts["lead_hour"].eq(1)
        & forecasts["forecast_time"].between(
            window_start,
            window_end,
            inclusive="both",
        )
    ].copy()

    event_forecasts = (
        event_forecasts.dropna(
            subset=[
                "issue_time",
                "forecast_time",
                "predicted_stage_ft",
            ]
        )
        .sort_values(["issue_time", "forecast_time"])
        .reset_index(drop=True)
    )

    if event_forecasts.empty:
        raise ValueError(
            f"No LSTM forecasts for event rank {event_rank} overlap "
            f"{window_start} through {window_end}."
        )

    return event_forecasts


# ---------------------------------------------------------------------
# Major-event identification
# ---------------------------------------------------------------------


def identify_threshold_events(
    stage: pd.DataFrame,
    threshold_ft: float,
) -> pd.DataFrame:
    """
    Identify continuous stage-threshold exceedance events.

    An event ends when:
    1. stage falls below the threshold, or
    2. the gap between observations exceeds MAX_GAP_MINUTES.
    """

    data = (
        stage.dropna(subset=["datetime", "gage_height_ft"])
        .sort_values("datetime")
        .copy()
    )

    if data.empty or pd.isna(threshold_ft):
        return pd.DataFrame()

    data["above_threshold"] = data["gage_height_ft"] >= threshold_ft

    time_gap_minutes = data["datetime"].diff().dt.total_seconds().div(60)

    previous_above = data["above_threshold"].shift(fill_value=False)

    data["new_event"] = data["above_threshold"] & (
        ~previous_above | time_gap_minutes.gt(MAX_GAP_MINUTES)
    )

    data["event_number"] = data["new_event"].cumsum()

    event_data = data.loc[data["above_threshold"]].copy()

    if event_data.empty:
        return pd.DataFrame()

    events = (
        event_data.groupby("event_number")
        .agg(
            start_time=("datetime", "min"),
            last_observation_time=("datetime", "max"),
            observation_count=("datetime", "size"),
            peak_stage_ft=("gage_height_ft", "max"),
            mean_stage_ft=("gage_height_ft", "mean"),
        )
        .reset_index(drop=True)
    )

    nominal_interval = pd.Timedelta(minutes=EXPECTED_INTERVAL_MINUTES)

    events["end_time"] = events["last_observation_time"] + nominal_interval

    events["duration_hours"] = (
        events["end_time"] - events["start_time"]
    ).dt.total_seconds() / 3600

    events["duration_days"] = events["duration_hours"] / 24

    events["threshold_ft"] = threshold_ft

    events = events.sort_values(
        "duration_hours",
        ascending=False,
    ).reset_index(drop=True)

    events["duration_rank"] = np.arange(1, len(events) + 1)

    return events


def get_ranked_major_events(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Identify and rank major events at the target gauge."""

    target = df.loc[df["site_id"].eq(TARGET_SITE_ID)].copy()

    stage = (
        target.loc[
            target["parameter_name"].eq("gage_height_ft"),
            [
                "datetime",
                "value",
                "major_flood_stage_ft",
            ],
        ]
        .rename(columns={"value": "gage_height_ft"})
        .copy()
    )

    threshold_values = stage["major_flood_stage_ft"].dropna()

    if threshold_values.empty:
        raise ValueError(
            f"No major-flood threshold is available "
            f"for target gauge {TARGET_SITE_ID}."
        )

    major_threshold_ft = float(threshold_values.iloc[0])

    events = identify_threshold_events(
        stage=stage,
        threshold_ft=major_threshold_ft,
    )

    if events.empty:
        raise ValueError(
            f"No major-stage exceedance events were found "
            f"for target gauge {TARGET_SITE_ID}."
        )

    events.insert(0, "site_id", TARGET_SITE_ID)
    events.insert(1, "threshold_name", "major")

    return events


def select_major_event(
    events: pd.DataFrame,
    event_rank: int,
) -> pd.Series:
    """Select a major event using its duration rank."""

    if event_rank < 1:
        raise ValueError("--event-rank must be at least 1.")

    available_ranks = events["duration_rank"].astype(int).tolist()

    if event_rank not in available_ranks:
        raise ValueError(
            f"Major event rank {event_rank} is not available. "
            f"Available ranks: {available_ranks}"
        )

    selected = events.loc[events["duration_rank"].eq(event_rank)].iloc[0]

    return selected


# ---------------------------------------------------------------------
# NWM retrospective data
# ---------------------------------------------------------------------


def find_nwm_file_for_window(
    nwm_directory: Path,
    site_id: str,
    reach_id: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> Path:
    """
    Find an NWM retrospective CSV that contains the complete
    selected event window.
    """

    pattern = (
        f"USGS_{site_id}_"
        f"ReachID_{reach_id}_"
        "NWM_v3_retrospective_*_"
        "streamflow_gauge_height.csv"
    )

    candidates = sorted(nwm_directory.glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            "No NWM retrospective file was found matching:\n"
            f"  {nwm_directory / pattern}\n\n"
            "Run 09_retrieve_nwm_retrospective_gauge_height.py "
            "for a period that contains the selected event."
        )

    containing_files: list[Path] = []

    for path in candidates:
        try:
            dates = pd.read_csv(
                path,
                usecols=["datetime"],
            )

            dates["datetime"] = pd.to_datetime(
                dates["datetime"],
                errors="coerce",
                utc=True,
            )

            available_start = dates["datetime"].min()

            available_end = dates["datetime"].max()

            if (
                pd.notna(available_start)
                and pd.notna(available_end)
                and available_start <= window_start
                and available_end >= window_end
            ):
                containing_files.append(path)

        except Exception as exc:
            print(f"Warning: could not inspect NWM file " f"{path.name}: {exc}")

    if not containing_files:
        raise FileNotFoundError(
            "NWM files were found, but none contains the full "
            "selected event window:\n"
            f"  {window_start} through {window_end}\n\n"
            "Retrieve a wider NWM period using Script 09."
        )

    # Prefer the smallest suitable file.
    return min(
        containing_files,
        key=lambda path: path.stat().st_size,
    )


def load_nwm_event_window(
    path: Path,
    site_id: str,
    reach_id: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Load hourly NWM streamflow and estimated gauge height for
    the selected event window.
    """

    if not path.exists():
        raise FileNotFoundError(f"NWM input file not found: {path}")

    required_columns = [
        "site_id",
        "datetime",
        "feature_id",
        "streamflow_cms",
        "streamflow_cfs",
        "estimated_gauge_height_ft",
        "within_usgs_rating_range",
    ]

    nwm = pd.read_csv(
        path,
        usecols=required_columns,
        dtype={"site_id": "string"},
        low_memory=False,
    )

    nwm["site_id"] = nwm["site_id"].astype("string").str.zfill(8)

    nwm["datetime"] = pd.to_datetime(
        nwm["datetime"],
        errors="coerce",
        utc=True,
    )

    nwm["feature_id"] = pd.to_numeric(
        nwm["feature_id"],
        errors="coerce",
    )

    for column in [
        "streamflow_cms",
        "streamflow_cfs",
        "estimated_gauge_height_ft",
    ]:
        nwm[column] = pd.to_numeric(
            nwm[column],
            errors="coerce",
        )

    nwm = nwm.dropna(subset=["datetime"])

    available_site_ids = sorted(nwm["site_id"].dropna().unique().tolist())
    available_reach_ids = sorted(
        nwm["feature_id"].dropna().astype(int).unique().tolist()
    )

    target_nwm = nwm.loc[
        nwm["site_id"].eq(site_id) & nwm["feature_id"].eq(reach_id)
    ].copy()

    if target_nwm.empty:
        raise ValueError(
            f"NWM file does not contain USGS {site_id} and "
            f"ReachID {reach_id}. Available site IDs: "
            f"{available_site_ids}; available ReachIDs: "
            f"{available_reach_ids}."
        )

    duplicate_mask = target_nwm["datetime"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(
            f"NWM file contains {duplicate_count} rows with "
            "duplicate timestamps for the target site and reach."
        )

    available_start = target_nwm["datetime"].min()
    available_end = target_nwm["datetime"].max()

    overlaps_window = available_start <= window_end and available_end >= window_start
    if not overlaps_window:
        raise ValueError(
            "The selected event window does not overlap the NWM "
            f"data. Event window: {window_start} through "
            f"{window_end}; NWM coverage: {available_start} "
            f"through {available_end}."
        )

    if available_start > window_start or available_end < window_end:
        raise ValueError(
            "The NWM file overlaps but does not cover the complete "
            f"buffered event window. Event window: {window_start} "
            f"through {window_end}; NWM coverage: "
            f"{available_start} through {available_end}. Retrieve "
            "a wider NWM period using Script 09."
        )

    nwm_event = (
        target_nwm.loc[
            target_nwm["datetime"].between(
                window_start,
                window_end,
                inclusive="both",
            )
        ]
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    if nwm_event.empty:
        raise ValueError(
            "No NWM values were found within the selected event " "window."
        )

    return nwm_event, available_start, available_end


def create_target_hourly_comparison(
    event_window: pd.DataFrame,
    nwm_event: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join hourly USGS observations and hourly NWM estimates at
    the target gauge.
    """

    target_observations = event_window.loc[
        event_window["site_id"].eq(TARGET_SITE_ID)
    ].copy()

    observed = (
        target_observations.pivot_table(
            index="datetime",
            columns="parameter_name",
            values="value",
            aggfunc="mean",
        )
        .sort_index()
        .rename_axis(columns=None)
    )

    for column in [
        "streamflow_cfs",
        "gage_height_ft",
    ]:
        if column not in observed.columns:
            observed[column] = np.nan

    observed_hourly = (
        observed[
            [
                "streamflow_cfs",
                "gage_height_ft",
            ]
        ]
        .resample("1h")
        .mean()
        .rename(
            columns={
                "streamflow_cfs": ("observed_streamflow_cfs"),
                "gage_height_ft": ("observed_gage_height_ft"),
            }
        )
    )

    nwm_hourly = (
        nwm_event[
            [
                "datetime",
                "streamflow_cfs",
                "estimated_gauge_height_ft",
                "within_usgs_rating_range",
            ]
        ]
        .set_index("datetime")
        .sort_index()
        .resample("1h")
        .agg(
            {
                "streamflow_cfs": "mean",
                "estimated_gauge_height_ft": "mean",
                "within_usgs_rating_range": "last",
            }
        )
        .rename(
            columns={
                "streamflow_cfs": ("nwm_streamflow_cfs"),
                "estimated_gauge_height_ft": ("nwm_estimated_gage_height_ft"),
            }
        )
    )

    comparison = observed_hourly.join(
        nwm_hourly,
        how="outer",
    )

    comparison.index.name = "datetime"

    comparison["streamflow_error_cfs"] = (
        comparison["nwm_streamflow_cfs"] - comparison["observed_streamflow_cfs"]
    )

    comparison["gage_height_error_ft"] = (
        comparison["nwm_estimated_gage_height_ft"]
        - comparison["observed_gage_height_ft"]
    )

    column_order = [
        "observed_streamflow_cfs",
        "nwm_streamflow_cfs",
        "observed_gage_height_ft",
        "nwm_estimated_gage_height_ft",
        "within_usgs_rating_range",
        "streamflow_error_cfs",
        "gage_height_error_ft",
    ]

    return comparison[column_order]


# ---------------------------------------------------------------------
# Time-series preparation
# ---------------------------------------------------------------------


def extract_event_window(
    df: pd.DataFrame,
    selected_event: pd.Series,
    days_before: float,
    days_after: float,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Extract network data around the selected event."""

    window_start = selected_event["start_time"] - pd.Timedelta(days=days_before)

    window_end = selected_event["end_time"] + pd.Timedelta(days=days_after)

    event_window = df.loc[
        df["datetime"].between(
            window_start,
            window_end,
            inclusive="both",
        )
    ].copy()

    if event_window.empty:
        raise ValueError(
            "No observations were found within the selected " "event window."
        )

    return event_window, window_start, window_end


def create_site_timeseries(
    group: pd.DataFrame,
    resample: str,
) -> pd.DataFrame:
    """Pivot streamflow and stage into time-indexed columns."""

    wide = (
        group.pivot_table(
            index="datetime",
            columns="parameter_name",
            values="value",
            aggfunc="last",
        )
        .sort_index()
        .rename_axis(columns=None)
    )

    for parameter in [
        "streamflow_cfs",
        "gage_height_ft",
    ]:
        if parameter not in wide.columns:
            wide[parameter] = np.nan

    wide = wide[
        [
            "streamflow_cfs",
            "gage_height_ft",
        ]
    ]

    if resample == "hourly":
        wide = wide.resample("1h").mean()

    return wide


def build_network_timeseries(
    event_window: pd.DataFrame,
    resample: str,
) -> dict[str, pd.DataFrame]:
    """Create one wide time-series dataframe per gauge."""

    site_series: dict[str, pd.DataFrame] = {}

    for site_id in NETWORK_SITE_IDS:
        group = event_window.loc[event_window["site_id"].eq(site_id)]

        if group.empty:
            continue

        site_series[site_id] = create_site_timeseries(
            group=group,
            resample=resample,
        )

    return site_series


# ---------------------------------------------------------------------
# Peak timing and lag analysis
# ---------------------------------------------------------------------


def compute_peak_summary(
    event_window: pd.DataFrame,
    site_series: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Find local streamflow and stage peaks for each gauge."""

    metadata = (
        event_window[
            [
                "site_id",
                "site_name",
                "river",
            ]
        ]
        .drop_duplicates("site_id")
        .set_index("site_id")
    )

    peak_rows: list[dict[str, object]] = []

    for site_id in NETWORK_SITE_IDS:
        if site_id not in site_series:
            continue

        series = site_series[site_id]

        flow = series["streamflow_cfs"].dropna()

        stage = series["gage_height_ft"].dropna()

        flow_peak_time = flow.idxmax() if not flow.empty else pd.NaT

        stage_peak_time = stage.idxmax() if not stage.empty else pd.NaT

        peak_rows.append(
            {
                "site_id": site_id,
                "site_name": (
                    metadata.loc[
                        site_id,
                        "site_name",
                    ]
                    if site_id in metadata.index
                    else ""
                ),
                "river": (
                    metadata.loc[
                        site_id,
                        "river",
                    ]
                    if site_id in metadata.index
                    else ""
                ),
                "peak_streamflow_cfs": (
                    float(flow.max()) if not flow.empty else np.nan
                ),
                "streamflow_peak_time": (flow_peak_time),
                "peak_stage_ft": (float(stage.max()) if not stage.empty else np.nan),
                "stage_peak_time": (stage_peak_time),
            }
        )

    peaks = pd.DataFrame(peak_rows)

    target_flow_peak = peaks.loc[
        peaks["site_id"].eq(TARGET_SITE_ID),
        "streamflow_peak_time",
    ]

    target_stage_peak = peaks.loc[
        peaks["site_id"].eq(TARGET_SITE_ID),
        "stage_peak_time",
    ]

    if target_flow_peak.empty:
        target_flow_time = pd.NaT
    else:
        target_flow_time = target_flow_peak.iloc[0]

    if target_stage_peak.empty:
        target_stage_time = pd.NaT
    else:
        target_stage_time = target_stage_peak.iloc[0]

    peaks["flow_peak_lag_to_target_hours"] = (
        target_flow_time - peaks["streamflow_peak_time"]
    ).dt.total_seconds() / 3600

    peaks["stage_peak_lag_to_target_hours"] = (
        target_stage_time - peaks["stage_peak_time"]
    ).dt.total_seconds() / 3600

    # Positive lag means the upstream/tributary gauge peaked before
    # the target gauge.
    peaks["network_order"] = peaks["site_id"].map(
        {site_id: index for index, site_id in enumerate(NETWORK_SITE_IDS)}
    )

    peaks = (
        peaks.sort_values("network_order")
        .drop(columns="network_order")
        .reset_index(drop=True)
    )

    return peaks


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------


def get_site_metadata(
    event_window: pd.DataFrame,
    site_id: str,
) -> pd.Series:
    """Return one metadata row for a gauge."""

    metadata = event_window.loc[event_window["site_id"].eq(site_id)]

    if metadata.empty:
        raise ValueError(f"No metadata found for site {site_id}.")

    return metadata.iloc[0]


def format_time_axis(axis: plt.Axes) -> None:
    """Apply readable datetime formatting."""

    locator = mdates.AutoDateLocator(
        minticks=5,
        maxticks=10,
    )

    formatter = mdates.ConciseDateFormatter(locator)

    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)


def plot_streamflow_propagation(
    site_series: dict[str, pd.DataFrame],
    target_comparison: pd.DataFrame,
    event_window: pd.DataFrame,
    peaks: pd.DataFrame,
    selected_event: pd.Series,
    event_rank: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    resample: str,
) -> Path:
    """Plot stacked streamflow hydrographs."""

    available_sites = [
        site_id for site_id in NETWORK_SITE_IDS if site_id in site_series
    ]

    figure, axes = plt.subplots(
        nrows=len(available_sites),
        ncols=1,
        figsize=(13, 2.4 * len(available_sites)),
        sharex=True,
    )

    axes = np.atleast_1d(axes)

    for axis, site_id in zip(
        axes,
        available_sites,
    ):
        series = site_series[site_id]
        metadata = get_site_metadata(
            event_window,
            site_id,
        )

        if site_id == TARGET_SITE_ID:
            observed_flow = target_comparison["observed_streamflow_cfs"].dropna()
            nwm_flow = target_comparison["nwm_streamflow_cfs"].dropna()

            axis.plot(
                observed_flow.index,
                observed_flow.values,
                linewidth=HYDROGRAPH_LINE_WIDTH,
                label="USGS observed (hourly)",
            )
            axis.plot(
                nwm_flow.index,
                nwm_flow.values,
                linewidth=HYDROGRAPH_LINE_WIDTH,
                linestyle="--",
                label="NWM retrospective (hourly)",
            )
        else:
            flow = series["streamflow_cfs"].dropna()
            axis.plot(
                flow.index,
                flow.values,
                linewidth=HYDROGRAPH_LINE_WIDTH,
            )

        peak_row = peaks.loc[peaks["site_id"].eq(site_id)]

        if not peak_row.empty:
            peak_time = peak_row["streamflow_peak_time"].iloc[0]

            peak_value = peak_row["peak_streamflow_cfs"].iloc[0]

            lag_hours = peak_row["flow_peak_lag_to_target_hours"].iloc[0]

            if pd.notna(peak_time):
                axis.axvline(
                    peak_time,
                    linestyle="--",
                    linewidth=1,
                )

                label_text = f"Peak: {peak_value:,.0f} cfs"

                if site_id != TARGET_SITE_ID and pd.notna(lag_hours):
                    label_text += f"\nLead to target: " f"{lag_hours:.1f} h"

                axis.text(
                    0.99,
                    0.93,
                    label_text,
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=12,
                )

        axis.axvspan(
            selected_event["start_time"],
            selected_event["end_time"],
            alpha=0.10,
        )

        axis.set_xlim(
            window_start,
            window_end,
        )

        axis.set_ylabel("Q (ft³/s)")

        axis.set_title(
            f"{metadata['site_name']} | " f"USGS {site_id}",
            loc="left",
            fontsize=12,
        )

        axis.grid(
            True,
            alpha=0.25,
        )

        axis.tick_params(axis="both", labelsize=12)

    format_time_axis(axes[-1])
    axes[-1].set_xlabel("Datetime (UTC)")

    target_name = get_site_metadata(
        event_window,
        TARGET_SITE_ID,
    )["site_name"]

    # figure.suptitle(
    #     "Flood-wave propagation: streamflow\n"
    #     f"Target: {target_name} | USGS {TARGET_SITE_ID} | "
    #     f"Major event rank {event_rank} by duration\n"
    #     f"Target event: "
    #     f"{selected_event['start_time']:%Y-%m-%d %H:%M} to "
    #     f"{selected_event['end_time']:%Y-%m-%d %H:%M} UTC",
    #     fontsize=14,
    # )

    # figure.tight_layout(
    #     rect=[0, 0, 1, 0.94]
    # )

    if TARGET_SITE_ID in available_sites:
        target_axis = axes[available_sites.index(TARGET_SITE_ID)]
        handles, labels = target_axis.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=2,
            fontsize=11,
            frameon=True,
        )

    figure.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = FIGURE_DIR / (
        f"{TARGET_SITE_ID}_major_event_rank_"
        f"{event_rank}_{resample}_"
        f"streamflow_propagation.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def plot_stage_propagation(
    site_series: dict[str, pd.DataFrame],
    target_comparison: pd.DataFrame,
    lstm_forecasts: pd.DataFrame,
    event_window: pd.DataFrame,
    peaks: pd.DataFrame,
    selected_event: pd.Series,
    event_rank: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    resample: str,
) -> Path:
    """Plot stacked stage hydrographs and flood thresholds."""

    available_sites = [
        site_id for site_id in NETWORK_SITE_IDS if site_id in site_series
    ]

    figure, axes = plt.subplots(
        nrows=len(available_sites),
        ncols=1,
        figsize=(13, 2.7 * len(available_sites)),
        sharex=True,
    )

    axes = np.atleast_1d(axes)

    for axis, site_id in zip(
        axes,
        available_sites,
    ):
        series = site_series[site_id]
        metadata = get_site_metadata(
            event_window,
            site_id,
        )

        if site_id == TARGET_SITE_ID:
            observed_stage = target_comparison["observed_gage_height_ft"].dropna()

            axis.plot(
                observed_stage.index,
                observed_stage.values,
                linewidth=HYDROGRAPH_LINE_WIDTH,
                label="USGS observed stage (hourly)",
            )

            axis.plot(
                lstm_forecasts["forecast_time"],
                lstm_forecasts["predicted_stage_ft"],
                color="#2CA02C",
                linewidth=1.5,
                alpha=0.85,
                label="LSTM first-step forecast",
            )
        else:
            stage = series["gage_height_ft"].dropna()
            axis.plot(
                stage.index,
                stage.values,
                linewidth=HYDROGRAPH_LINE_WIDTH,
                label="Gage height",
            )

        for threshold_name, column in THRESHOLD_COLUMNS.items():
            threshold_ft = metadata[column]

            if pd.isna(threshold_ft):
                continue

            axis.axhline(
                threshold_ft,
                color=THRESHOLD_COLORS[threshold_name],
                linestyle=THRESHOLD_LINE_STYLES[threshold_name],
                linewidth=THRESHOLD_LINE_WIDTH,
                label=threshold_name.title(),
            )

        peak_row = peaks.loc[peaks["site_id"].eq(site_id)]

        if not peak_row.empty:
            peak_time = peak_row["stage_peak_time"].iloc[0]

            peak_value = peak_row["peak_stage_ft"].iloc[0]

            lag_hours = peak_row["stage_peak_lag_to_target_hours"].iloc[0]

            if pd.notna(peak_time):
                axis.axvline(
                    peak_time,
                    linestyle="--",
                    linewidth=1,
                )

                label_text = f"Peak: {peak_value:.2f} ft"

                if site_id != TARGET_SITE_ID and pd.notna(lag_hours):
                    label_text += f"\nLead to target: " f"{lag_hours:.1f} h"

                axis.text(
                    0.99,
                    0.93,
                    label_text,
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=12,
                )

        axis.axvspan(
            selected_event["start_time"],
            selected_event["end_time"],
            alpha=0.10,
        )

        axis.set_xlim(
            window_start,
            window_end,
        )

        axis.set_ylabel("Stage (ft)")

        axis.set_title(
            f"{metadata['site_name']} | " f"USGS {site_id}",
            loc="left",
            fontsize=12,
        )

        axis.grid(
            True,
            alpha=0.25,
        )

        axis.tick_params(axis="both", labelsize=12)

    format_time_axis(axes[-1])
    axes[-1].set_xlabel("Datetime (UTC)")

    target_name = get_site_metadata(
        event_window,
        TARGET_SITE_ID,
    )["site_name"]

    # figure.suptitle(
    #     "Flood-wave propagation: gage height\n"
    #     f"Target: {target_name} | USGS {TARGET_SITE_ID} | "
    #     f"Major event rank {event_rank} by duration\n"
    #     f"Target event: "
    #     f"{selected_event['start_time']:%Y-%m-%d %H:%M} to "
    #     f"{selected_event['end_time']:%Y-%m-%d %H:%M} UTC",
    #     fontsize=14,
    # )

    # figure.tight_layout(
    #     rect=[0, 0, 1, 0.94]
    # )

    if TARGET_SITE_ID in available_sites:
        target_axis = axes[available_sites.index(TARGET_SITE_ID)]
        handles, labels = target_axis.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=len(labels),
            fontsize=10,
            frameon=True,
        )

    figure.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = FIGURE_DIR / (
        f"{TARGET_SITE_ID}_major_event_rank_"
        f"{event_rank}_{resample}_"
        f"stage_propagation.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path

def create_lstm_prediction_request(
    major_events: pd.DataFrame,
    days_before: float,
    days_after: float,
) -> pd.DataFrame:
    """
    Create complete UTC-day prediction windows for LSTM evaluation.

    The requested start is floored to 00:00 UTC.
    The requested end is ceiled to 00:00 UTC of the next day when needed.
    """

    request = major_events[
        [
            "duration_rank",
            "start_time",
            "end_time",
            "duration_hours",
            "peak_stage_ft",
        ]
    ].copy()

    request = request.rename(
        columns={
            "duration_rank": "event_rank",
            "start_time": "major_event_start_utc",
            "end_time": "major_event_end_utc",
            "duration_hours": "major_event_duration_hours",
            "peak_stage_ft": "observed_peak_stage_ft",
        }
    )

    buffered_start = (
        request["major_event_start_utc"]
        - pd.to_timedelta(days_before, unit="D")
    )

    buffered_end = (
        request["major_event_end_utc"]
        + pd.to_timedelta(days_after, unit="D")
    )

    request["requested_prediction_start_utc"] = buffered_start.dt.floor("D")
    request["requested_prediction_end_utc"] = buffered_end.dt.ceil("D")

    request["target_site_id"] = TARGET_SITE_ID
    request["forecast_frequency_hours"] = 12
    request["forecast_horizon_hours"] = 12
    request["prediction_timestep_hours"] = 1
    request["input_history_hours"] = 12

    column_order = [
        "event_rank",
        "target_site_id",
        "major_event_start_utc",
        "major_event_end_utc",
        "major_event_duration_hours",
        "observed_peak_stage_ft",
        "requested_prediction_start_utc",
        "requested_prediction_end_utc",
        "input_history_hours",
        "forecast_frequency_hours",
        "forecast_horizon_hours",
        "prediction_timestep_hours",
    ]

    return request[column_order]
# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.days_before < 0:
        raise ValueError("--days-before cannot be negative.")

    if args.days_after < 0:
        raise ValueError("--days-after cannot be negative.")

    print(f"Reading: {INPUT_FILE}")

    df = load_network_data(INPUT_FILE)

    major_events = get_ranked_major_events(df)

    print(
        "\nMajor-stage events at target gauge "
        f"{TARGET_SITE_ID}, ranked longest to shortest:"
    )

    lstm_prediction_request = create_lstm_prediction_request(
        major_events=major_events,
        days_before=args.days_before,
        days_after=args.days_after,)

    print(
        major_events[
            [
                "duration_rank",
                "start_time",
                "end_time",
                "duration_hours",
                "duration_days",
                "peak_stage_ft",
            ]
        ].to_string(
            index=False,
        )
    )

    selected_event = select_major_event(
        events=major_events,
        event_rank=args.event_rank,
    )

    print(f"\nSelected major event rank: " f"{args.event_rank}")

    print(f"Start:    " f"{selected_event['start_time']}")

    print(f"End:      " f"{selected_event['end_time']}")

    print(
        f"Duration: "
        f"{selected_event['duration_hours']:.2f} hours "
        f"({selected_event['duration_days']:.2f} days)"
    )

    print(f"Peak stage at target: " f"{selected_event['peak_stage_ft']:.2f} ft")

    event_window, window_start, window_end = extract_event_window(
        df=df,
        selected_event=selected_event,
        days_before=args.days_before,
        days_after=args.days_after,
    )

    lstm_forecast_file = args.lstm_forecast_file.resolve()
    lstm_forecasts = load_lstm_event_forecasts(
        path=lstm_forecast_file,
        event_rank=args.event_rank,
        window_start=window_start,
        window_end=window_end,
    )

    print(f"\nSelected LSTM forecast file: {lstm_forecast_file}")
    print(
        "LSTM forecasts for selected event:\n"
        f"  Issue times:              "
        f"{lstm_forecasts['issue_time'].nunique():,}\n"
        f"  Forecast points:          {len(lstm_forecasts):,}\n"
        f"  Forecast-time coverage:   "
        f"{lstm_forecasts['forecast_time'].min()} through "
        f"{lstm_forecasts['forecast_time'].max()}"
    )

    if args.nwm_file is None:
        nwm_file = find_nwm_file_for_window(
            nwm_directory=NWM_INPUT_DIR,
            site_id=TARGET_SITE_ID,
            reach_id=TARGET_REACH_ID,
            window_start=window_start,
            window_end=window_end,
        )
    else:
        nwm_file = args.nwm_file.resolve()

    print(f"\nSelected NWM file: {nwm_file}")
    print(f"Selected event window: {window_start} through " f"{window_end}")

    (
        nwm_event,
        nwm_available_start,
        nwm_available_end,
    ) = load_nwm_event_window(
        path=nwm_file,
        site_id=TARGET_SITE_ID,
        reach_id=TARGET_REACH_ID,
        window_start=window_start,
        window_end=window_end,
    )

    print(f"Available NWM data: {nwm_available_start} through " f"{nwm_available_end}")

    target_comparison = create_target_hourly_comparison(
        event_window=event_window,
        nwm_event=nwm_event,
    )

    observed_mask = (
        target_comparison[
            [
                "observed_streamflow_cfs",
                "observed_gage_height_ft",
            ]
        ]
        .notna()
        .any(axis=1)
    )
    nwm_mask = (
        target_comparison[
            [
                "nwm_streamflow_cfs",
                "nwm_estimated_gage_height_ft",
            ]
        ]
        .notna()
        .any(axis=1)
    )

    print(
        "Hourly target-gauge coverage:\n"
        f"  USGS observations:       {int(observed_mask.sum()):,}\n"
        f"  NWM values:              {int(nwm_mask.sum()):,}\n"
        f"  Matched timestamps:      "
        f"{int((observed_mask & nwm_mask).sum()):,}"
    )

    site_series = build_network_timeseries(
        event_window=event_window,
        resample=args.resample,
    )

    peaks = compute_peak_summary(
        event_window=event_window,
        site_series=site_series,
    )

    print("\nPeak timing and lag summary:")
    print(
        peaks.to_string(
            index=False,
        )
    )

    output_tag = (
        f"{TARGET_SITE_ID}_major_event_" f"rank_{args.event_rank}_{args.resample}"
    )

    ranked_events_file = OUTPUT_DIR / f"{TARGET_SITE_ID}_ranked_major_events.csv"

    selected_event_file = OUTPUT_DIR / f"{output_tag}_selected_event.csv"

    event_window_file = OUTPUT_DIR / f"{output_tag}_network_timeseries.csv"

    peak_summary_file = OUTPUT_DIR / f"{output_tag}_peak_summary.csv"

    target_comparison_file = (
        OUTPUT_DIR / f"{output_tag}_target_observed_vs_nwm_hourly.csv"
    )

    lstm_request_file = (
        OUTPUT_DIR
        / f"{TARGET_SITE_ID}_lstm_event_prediction_requests.csv"
    )

    lstm_prediction_request.to_csv(
        lstm_request_file,
        index=False,
    )

    print("\nLSTM prediction periods:")
    print(
        lstm_prediction_request.to_string(
            index=False,
        )
    )
    major_events.to_csv(
        ranked_events_file,
        index=False,
    )

    pd.DataFrame([selected_event.to_dict()]).to_csv(
        selected_event_file,
        index=False,
    )

    event_window.to_csv(
        event_window_file,
        index=False,
    )

    peaks.to_csv(
        peak_summary_file,
        index=False,
    )

    target_comparison.reset_index().to_csv(
        target_comparison_file,
        index=False,
    )

    streamflow_figure = plot_streamflow_propagation(
        site_series=site_series,
        target_comparison=target_comparison,
        event_window=event_window,
        peaks=peaks,
        selected_event=selected_event,
        event_rank=args.event_rank,
        window_start=window_start,
        window_end=window_end,
        resample=args.resample,
    )

    stage_figure = plot_stage_propagation(
        site_series=site_series,
        target_comparison=target_comparison,
        lstm_forecasts=lstm_forecasts,
        event_window=event_window,
        peaks=peaks,
        selected_event=selected_event,
        event_rank=args.event_rank,
        window_start=window_start,
        window_end=window_end,
        resample=args.resample,
    )

    print("\nOutputs created:")
    print(f"  Ranked major events: {ranked_events_file}")
    print(f"  Selected event:      {selected_event_file}")
    print(f"  Network time series: {event_window_file}")
    print(f"  Peak summary:        {peak_summary_file}")
    print(f"  USGS/NWM comparison: {target_comparison_file}")
    print(f"  Streamflow figure:   {streamflow_figure}")
    print(f"  Stage figure:        {stage_figure}")
    print(f"  LSTM prediction request: {lstm_request_file}")


if __name__ == "__main__":
    main()
