#!/usr/bin/env python3
# coding: utf-8

"""
Download matching USGS gauge-height observations and compare them with
an operational NWM analysis-plus-short-range forecast CSV.

The NWM input is created by scripts/RetrieveNWM.py. When --nwm-file is
omitted, this script selects the newest matching operational CSV.

Example:

    python scripts/11_plot_operational_nwm_usgs.py

    python scripts/11_plot_operational_nwm_usgs.py \
        --nwm-file data/processed/nwm/operational_gauge_height/\
USGS_08210000_ReachID_3168766_NWM_analysis_7d_short_range_18h_\
2026073019_streamflow_gauge_height.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NWM_DIR = (
    ROOT / "data" / "processed" / "nwm" / "operational_gauge_height"
)
DEFAULT_OUTPUT_DIR = DEFAULT_NWM_DIR
DEFAULT_FIGURE_DIR = ROOT / "docs" / "figures" / "nwm_operational"
DEFAULT_LSTM_FILE = (
    ROOT
    / "data"
    / "processed"
    / "major_event_forecast_results_5_event (1).csv"
)
MASTER_METADATA_PATH = (
    ROOT / "data" / "processed" / "master_gauge_metadata.csv"
)

USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
GAGE_HEIGHT_PARAMETER_CODE = "00065"
REQUEST_TIMEOUT_SECONDS = 120
MAX_RETRIES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download matching USGS observations and plot them against "
            "operational NWM-derived gauge height."
        )
    )
    parser.add_argument(
        "--nwm-file",
        type=Path,
        default=None,
        help=(
            "Operational NWM CSV from RetrieveNWM.py. By default, the "
            "newest matching file is selected."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for downloaded observations and comparison CSV.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory for the comparison figure.",
    )
    parser.add_argument(
        "--lstm-file",
        type=Path,
        default=DEFAULT_LSTM_FILE,
        help=(
            "CSV containing LSTM forecasts. Default: "
            "data/processed/major_event_forecast_results_5_event (1).csv"
        ),
    )
    parser.add_argument(
        "--lstm-event-rank",
        type=int,
        default=5,
        help="Event rank to select from the LSTM file. Default: 5",
    )
    return parser.parse_args()


def find_latest_nwm_file(directory: Path) -> Path:
    pattern = (
        "USGS_*_ReachID_*_NWM_analysis_*d_short_range_*h_*_"
        "streamflow_gauge_height.csv"
    )
    candidates = list(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No operational NWM CSV was found matching:\n"
            f"  {directory / pattern}\n"
            "Run scripts/RetrieveNWM.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def normalize_site_id(value: object) -> str:
    cleaned = str(value).strip().replace(".0", "")
    if not cleaned.isdigit():
        raise ValueError(f"Invalid USGS site ID: {value!r}")
    return cleaned.zfill(8)


def load_nwm(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"NWM file not found: {path}")

    nwm = pd.read_csv(path, dtype={"site_id": "string"})
    required = {
        "site_id",
        "datetime",
        "data_type",
        "init_time",
        "estimated_gauge_height_ft",
    }
    missing = required.difference(nwm.columns)
    if missing:
        raise ValueError(
            f"NWM file is missing required columns: {sorted(missing)}"
        )

    nwm["site_id"] = nwm["site_id"].map(normalize_site_id)
    site_ids = nwm["site_id"].dropna().unique().tolist()
    if len(site_ids) != 1:
        raise ValueError(
            f"Expected one site_id in NWM file; found {site_ids}."
        )
    site_id = site_ids[0]

    for column in ["datetime", "init_time"]:
        nwm[column] = pd.to_datetime(
            nwm[column], errors="coerce", utc=True
        )
    nwm["estimated_gauge_height_ft"] = pd.to_numeric(
        nwm["estimated_gauge_height_ft"], errors="coerce"
    )
    nwm = (
        nwm.dropna(subset=["datetime", "estimated_gauge_height_ft"])
        .drop_duplicates("datetime", keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    if nwm.empty:
        raise ValueError("NWM file contains no valid gauge-height rows.")
    return nwm, site_id


def load_lstm_forecasts(
    path: Path,
    event_rank: int,
    latest_forecast_steps: int = 18,
) -> pd.DataFrame:
    """
    Load first-step predictions from earlier issues and the complete
    requested trajectory from the latest LSTM forecast issuance.
    """

    if not path.exists():
        raise FileNotFoundError(f"LSTM forecast file not found: {path}")

    required = {
        "event_rank",
        "issue_time",
        "lead_hour",
        "forecast_time",
        "predicted_stage_ft",
    }
    forecasts = pd.read_csv(path)
    missing = required.difference(forecasts.columns)
    if missing:
        raise ValueError(
            f"LSTM file is missing required columns: {sorted(missing)}"
        )

    forecasts["event_rank"] = pd.to_numeric(
        forecasts["event_rank"], errors="coerce"
    )
    forecasts["lead_hour"] = pd.to_numeric(
        forecasts["lead_hour"], errors="coerce"
    )
    forecasts["forecast_time"] = pd.to_datetime(
        forecasts["forecast_time"], errors="coerce", utc=True
    )
    forecasts["issue_time"] = pd.to_datetime(
        forecasts["issue_time"], errors="coerce", utc=True
    )
    forecasts["predicted_stage_ft"] = pd.to_numeric(
        forecasts["predicted_stage_ft"], errors="coerce"
    )

    available_ranks = sorted(
        forecasts["event_rank"].dropna().astype(int).unique().tolist()
    )
    if event_rank not in available_ranks:
        raise ValueError(
            f"LSTM event rank {event_rank} is unavailable. "
            f"Available ranks: {available_ranks}"
        )

    event_forecasts = forecasts.loc[
        forecasts["event_rank"].eq(event_rank)
    ].dropna(subset=["issue_time"])
    latest_issue_time = event_forecasts["issue_time"].max()

    previous_first_steps = (
        event_forecasts.loc[
            event_forecasts["issue_time"].lt(latest_issue_time)
            & event_forecasts["lead_hour"].eq(1)
        ]
        .dropna(
            subset=["forecast_time", "predicted_stage_ft", "lead_hour"]
        )
        .copy()
    )

    latest_forecast = (
        event_forecasts.loc[
            event_forecasts["issue_time"].eq(latest_issue_time)
            & event_forecasts["lead_hour"].between(
                1,
                latest_forecast_steps,
                inclusive="both",
            )
        ]
        .dropna(
            subset=["forecast_time", "predicted_stage_ft", "lead_hour"]
        )
        .drop_duplicates("forecast_time", keep="last")
        .sort_values("lead_hour")
        .reset_index(drop=True)
    )
    if len(latest_forecast) != latest_forecast_steps:
        raise ValueError(
            f"Expected {latest_forecast_steps} LSTM forecast steps for "
            f"event rank "
            f"{event_rank} at issue time {latest_issue_time}; "
            f"found {len(latest_forecast)}."
        )

    return (
        pd.concat(
            [previous_first_steps, latest_forecast],
            ignore_index=True,
        )
        .drop_duplicates("forecast_time", keep="last")
        .sort_values("forecast_time")
        .reset_index(drop=True)
    )


def request_usgs_gauge_height(
    site_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    params = {
        "format": "json",
        "sites": site_id,
        "parameterCd": GAGE_HEIGHT_PARAMETER_CODE,
        "startDT": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endDT": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "siteStatus": "all",
    }

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                USGS_IV_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            print(
                f"USGS request attempt {attempt}/{MAX_RETRIES} "
                f"failed: {exc}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)

    raise RuntimeError(
        "USGS request failed after all retry attempts."
    ) from last_error


def parse_usgs_gauge_height(
    payload: dict[str, Any],
    expected_site_id: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    series_list = payload.get("value", {}).get("timeSeries", [])

    for series in series_list:
        source = series.get("sourceInfo", {})
        site_codes = source.get("siteCode", [])
        if not site_codes:
            continue
        site_id = normalize_site_id(site_codes[0].get("value", ""))
        if site_id != expected_site_id:
            continue

        variable_codes = (
            series.get("variable", {}).get("variableCode", [])
        )
        if not variable_codes:
            continue
        parameter_code = str(
            variable_codes[0].get("value", "")
        ).zfill(5)
        if parameter_code != GAGE_HEIGHT_PARAMETER_CODE:
            continue

        for block in series.get("values", []):
            for observation in block.get("value", []):
                records.append(
                    {
                        "site_id": site_id,
                        "datetime": observation.get("dateTime"),
                        "observed_gauge_height_ft": observation.get("value"),
                        "qualifiers": ",".join(
                            observation.get("qualifiers", [])
                        ),
                    }
                )

    observations = pd.DataFrame(records)
    if observations.empty:
        raise RuntimeError(
            f"USGS returned no gauge-height observations for {expected_site_id}."
        )

    observations["datetime"] = pd.to_datetime(
        observations["datetime"], errors="coerce", utc=True
    )
    observations["observed_gauge_height_ft"] = pd.to_numeric(
        observations["observed_gauge_height_ft"], errors="coerce"
    )
    return (
        observations.dropna(
            subset=["datetime", "observed_gauge_height_ft"]
        )
        .drop_duplicates("datetime", keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )


def create_hourly_comparison(
    nwm: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    observed_hourly = (
        observations.set_index("datetime")["observed_gauge_height_ft"]
        .resample("1h")
        .mean()
    )

    comparison = nwm[
        [
            "datetime",
            "data_type",
            "init_time",
            "estimated_gauge_height_ft",
        ]
    ].copy()
    comparison = comparison.join(
        observed_hourly,
        on="datetime",
    )
    comparison["gauge_height_error_ft"] = (
        comparison["estimated_gauge_height_ft"]
        - comparison["observed_gauge_height_ft"]
    )
    comparison["absolute_error_ft"] = (
        comparison["gauge_height_error_ft"].abs()
    )
    comparison["squared_error_ft2"] = (
        comparison["gauge_height_error_ft"] ** 2
    )
    return comparison


def compute_metrics(comparison: pd.DataFrame) -> pd.DataFrame:
    matched = comparison.dropna(
        subset=[
            "estimated_gauge_height_ft",
            "observed_gauge_height_ft",
        ]
    )
    rows: list[dict[str, object]] = []

    for data_type, group in matched.groupby("data_type", sort=False):
        rows.append(
            {
                "data_type": data_type,
                "matched_hours": len(group),
                "bias_ft": group["gauge_height_error_ft"].mean(),
                "mae_ft": group["absolute_error_ft"].mean(),
                "rmse_ft": np.sqrt(group["squared_error_ft2"].mean()),
                "correlation": group[
                    [
                        "estimated_gauge_height_ft",
                        "observed_gauge_height_ft",
                    ]
                ].corr().iloc[0, 1],
            }
        )
    return pd.DataFrame(rows)


def load_thresholds(site_id: str) -> dict[str, float]:
    if not MASTER_METADATA_PATH.exists():
        return {}

    metadata = pd.read_csv(
        MASTER_METADATA_PATH,
        dtype={"site_id": "string"},
    )
    metadata["site_id"] = (
        metadata["site_id"].astype("string").str.zfill(8)
    )
    selected = metadata.loc[metadata["site_id"].eq(site_id)]
    if selected.empty:
        return {}

    row = selected.iloc[0]
    thresholds: dict[str, float] = {}
    for name, column in {
        "Action": "action_stage_ft",
        "Minor": "minor_flood_stage_ft",
        "Moderate": "moderate_flood_stage_ft",
        "Major": "major_flood_stage_ft",
    }.items():
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value):
            thresholds[name] = float(value)
    return thresholds


def plot_comparison(
    comparison: pd.DataFrame,
    observations: pd.DataFrame,
    lstm: pd.DataFrame,
    thresholds: dict[str, float],
    site_id: str,
    output_path: Path,
) -> None:
    figure, stage_axis = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(13, 6.5),
    )

    stage_axis.plot(
        observations["datetime"],
        observations["observed_gauge_height_ft"],
        color="#1F77B4",
        linewidth=1.6,
        alpha=0.55,
        label="USGS observed (instantaneous)",
    )

    styles = {
        "short_range": {
            "color": "#D62728",
            "linestyle": "--",
            "label": "NWM short-range forecast",
        }
    }
    for data_type, style in styles.items():
        subset = comparison.loc[
            comparison["data_type"].eq(data_type)
            & comparison["datetime"].le(lstm["forecast_time"].max())
        ]
        if subset.empty:
            continue
        stage_axis.plot(
            subset["datetime"],
            subset["estimated_gauge_height_ft"],
            linewidth=2.2,
            **style,
        )

    stage_axis.plot(
        lstm["forecast_time"],
        lstm["predicted_stage_ft"],
        color="#2CA02C",
        linewidth=2.2,
        label="LSTM forecast",
    )

    threshold_colors = {
        "Action": "#C7B700",
        "Minor": "#FF9800",
        "Moderate": "#E31A1C",
        "Major": "#C000C0",
    }
    for name, value in thresholds.items():
        stage_axis.axhline(
            value,
            color=threshold_colors[name],
            linewidth=0.9,
            label=f"{name} stage ({value:g} ft)",
        )

    forecast = comparison.loc[
        comparison["data_type"].eq("short_range")
        & comparison["datetime"].le(lstm["forecast_time"].max())
    ]
    if not forecast.empty:
        forecast_init = forecast["init_time"].dropna().iloc[0]
        forecast_end = comparison["datetime"].max()
        stage_axis.axvline(
            forecast_init,
            color="0.25",
            linestyle=":",
            linewidth=1.2,
        )
        stage_axis.axvspan(
            forecast_init,
            forecast_end,
            color="#D62728",
            alpha=0.06,
        )

    stage_axis.set_xlabel("Datetime (UTC)")
    stage_axis.set_ylabel("Gauge height (ft)")
    stage_axis.set_title(
        f"USGS {site_id}: observed, LSTM, and NWM gauge height",
        loc="left",
    )

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    stage_axis.xaxis.set_major_locator(locator)
    stage_axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator)
    )
    stage_axis.grid(True, alpha=0.25)
    stage_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=4,
        fontsize=12,
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    nwm_file = (
        args.nwm_file.resolve()
        if args.nwm_file
        else find_latest_nwm_file(DEFAULT_NWM_DIR)
    )
    lstm_file = args.lstm_file.resolve()
    output_dir = args.output_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading NWM data: {nwm_file}")
    nwm, site_id = load_nwm(nwm_file)
    print(
        f"Reading LSTM event rank {args.lstm_event_rank}: {lstm_file}"
    )
    lstm = load_lstm_forecasts(
        path=lstm_file,
        event_rank=args.lstm_event_rank,
        latest_forecast_steps=18,
    )
    start = nwm["datetime"].min()
    requested_end = nwm["datetime"].max()
    current_utc = pd.Timestamp.now(tz="UTC")
    observation_end = min(requested_end, current_utc)

    print(
        f"Requesting USGS {site_id} gauge height from "
        f"{start} through {observation_end}..."
    )
    payload = request_usgs_gauge_height(
        site_id=site_id,
        start=start,
        end=observation_end,
    )
    observations = parse_usgs_gauge_height(payload, site_id)
    comparison = create_hourly_comparison(nwm, observations)
    metrics = compute_metrics(
        comparison.loc[
            comparison["data_type"].eq("short_range")
            & comparison["datetime"].le(lstm["forecast_time"].max())
        ]
    )
    thresholds = load_thresholds(site_id)

    tag = nwm_file.stem.replace("_streamflow_gauge_height", "")
    observations_path = output_dir / f"{tag}_usgs_observations.csv"
    comparison_path = output_dir / f"{tag}_nwm_usgs_comparison.csv"
    metrics_path = output_dir / f"{tag}_evaluation_metrics.csv"
    figure_path = figure_dir / f"{tag}_nwm_usgs_gauge_height.png"

    observations.to_csv(observations_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    plot_comparison(
        comparison=comparison,
        observations=observations,
        lstm=lstm,
        thresholds=thresholds,
        site_id=site_id,
        output_path=figure_path,
    )

    print("\nEvaluation metrics:")
    print(metrics.to_string(index=False))
    print("\nOutputs created:")
    print(f"  USGS observations: {observations_path}")
    print(f"  Comparison CSV:    {comparison_path}")
    print(f"  Metrics CSV:       {metrics_path}")
    print(f"  Figure:            {figure_path}")


if __name__ == "__main__":
    main()
