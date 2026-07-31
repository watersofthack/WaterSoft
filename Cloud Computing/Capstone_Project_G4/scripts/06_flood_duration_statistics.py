#!/usr/bin/env python3
# coding: utf-8

"""
Identify continuous flood-stage exceedance events from original
15-minute USGS gage-height observations.

Example:

    python scripts/06_flood_duration_statistics.py \
        --site-id 08210000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "raw"
    / "usgs"
    / "timeseries"
    / "usgs_gauge_height_streamflow_long.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "flood_duration"
FIGURE_DIR = ROOT / "docs" / "figures" / "flood_duration"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_INTERVAL_MINUTES = 15
MAX_GAP_MINUTES = 30

THRESHOLD_COLUMNS = {
    "action": "action_stage_ft",
    "minor": "minor_flood_stage_ft",
    "moderate": "moderate_flood_stage_ft",
    "major": "major_flood_stage_ft",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute continuous flood-stage exceedance durations."
    )

    parser.add_argument(
        "--site-id",
        type=str,
        default="08210000",
        help="Eight-digit USGS site ID or 'all'.",
    )

    return parser.parse_args()


def load_stage_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    required = [
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
        usecols=required,
        parse_dates=["datetime"],
        dtype={"site_id": "string"},
        low_memory=False,
    )

    df["site_id"] = df["site_id"].str.zfill(8)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    for column in THRESHOLD_COLUMNS.values():
        df[column] = pd.to_numeric(df[column], errors="coerce")

    stage = (
        df.loc[df["parameter_name"].eq("gage_height_ft")]
        .rename(columns={"value": "gage_height_ft"})
        .drop_duplicates(["site_id", "datetime"], keep="last")
        .sort_values(["site_id", "datetime"])
        .reset_index(drop=True)
    )

    return stage


def select_sites(df: pd.DataFrame, requested: str) -> pd.DataFrame:
    if requested.lower() == "all":
        return df.copy()

    site_id = requested.zfill(8)
    available = sorted(df["site_id"].dropna().unique())

    if site_id not in available:
        raise ValueError(
            f"Site {site_id} not found. Available sites: "
            + ", ".join(available)
        )

    return df.loc[df["site_id"].eq(site_id)].copy()


def identify_events(
    group: pd.DataFrame,
    threshold_ft: float,
    threshold_name: str,
) -> pd.DataFrame:
    """
    Identify consecutive stage observations at or above a threshold.

    Events are broken when:
    1. stage falls below the threshold, or
    2. the observation gap exceeds MAX_GAP_MINUTES.
    """

    data = group.dropna(
        subset=["datetime", "gage_height_ft"]
    ).copy()

    if data.empty or pd.isna(threshold_ft):
        return pd.DataFrame()

    data = data.sort_values("datetime")

    data["above_threshold"] = (
        data["gage_height_ft"] >= threshold_ft
    )

    time_gap_minutes = (
        data["datetime"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    new_event = (
        data["above_threshold"]
        & (
            ~data["above_threshold"].shift(fill_value=False)
            | time_gap_minutes.gt(MAX_GAP_MINUTES)
        )
    )

    data["event_number"] = new_event.cumsum()

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

    nominal_interval = pd.Timedelta(
        minutes=EXPECTED_INTERVAL_MINUTES
    )

    events["end_time"] = (
        events["last_observation_time"] + nominal_interval
    )

    events["duration_hours"] = (
        events["end_time"] - events["start_time"]
    ).dt.total_seconds() / 3600

    events["duration_days"] = (
        events["duration_hours"] / 24
    )

    events["threshold_name"] = threshold_name
    events["threshold_ft"] = threshold_ft

    return events


def plot_event_durations(
    events: pd.DataFrame,
    site_id: str,
    site_name: str,
) -> None:
    """Plot ranked event durations for all flood-stage thresholds."""

    if events.empty:
        return

    figure, axis = plt.subplots(figsize=(9, 5))

    for threshold_name, group in events.groupby(
        "threshold_name",
        sort=False,
    ):
        durations = np.sort(
            group["duration_hours"].to_numpy()
        )

        event_ranks = np.arange(
            1,
            len(durations) + 1,
        )

    threshold_order = [
        "action",
        "minor",
        "moderate",
        "major",
    ]

    for threshold_name in threshold_order:
        group = events.loc[
            events["threshold_name"].eq(threshold_name)
        ]

        if group.empty:
            continue

        durations = np.sort(
            group["duration_hours"].to_numpy()
        )

        event_ranks = np.arange(
            1,
            len(durations) + 1,
        )

        axis.scatter(
            event_ranks,
            durations,
            s=35,
            label=threshold_name.title(),
        )

    axis.set_yscale("log")
    axis.set_xlabel("Event rank, shortest to longest")
    axis.set_ylabel("Duration (hours)")
    axis.set_title(
        f"Flood-stage exceedance durations\n"
        f"{site_name} | USGS {site_id}"
    )
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(title="Threshold")

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIR / f"{site_id}_flood_event_durations.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    args = parse_args()

    stage = load_stage_data(INPUT_FILE)
    stage = select_sites(stage, args.site_id)

    all_events = []
    summary_rows = []

    for site_id, group in stage.groupby("site_id"):
        site_name = group["site_name"].iloc[0]
        river = group["river"].iloc[0]

        site_events = []

        for threshold_name, column in THRESHOLD_COLUMNS.items():
            threshold_values = group[column].dropna()

            if threshold_values.empty:
                continue

            threshold_ft = float(threshold_values.iloc[0])

            events = identify_events(
                group=group,
                threshold_ft=threshold_ft,
                threshold_name=threshold_name,
            )

            if events.empty:
                summary_rows.append(
                    {
                        "site_id": site_id,
                        "site_name": site_name,
                        "river": river,
                        "threshold_name": threshold_name,
                        "threshold_ft": threshold_ft,
                        "event_count": 0,
                        "total_hours": 0,
                        "mean_duration_hours": np.nan,
                        "median_duration_hours": np.nan,
                        "maximum_duration_hours": np.nan,
                    }
                )
                continue

            events.insert(0, "site_id", site_id)
            events.insert(1, "site_name", site_name)
            events.insert(2, "river", river)

            all_events.append(events)
            site_events.append(events)

            summary_rows.append(
                {
                    "site_id": site_id,
                    "site_name": site_name,
                    "river": river,
                    "threshold_name": threshold_name,
                    "threshold_ft": threshold_ft,
                    "event_count": len(events),
                    "total_hours": events["duration_hours"].sum(),
                    "mean_duration_hours": events[
                        "duration_hours"
                    ].mean(),
                    "median_duration_hours": events[
                        "duration_hours"
                    ].median(),
                    "maximum_duration_hours": events[
                        "duration_hours"
                    ].max(),
                }
            )

        if site_events:
            plot_event_durations(
                events=pd.concat(site_events, ignore_index=True),
                site_id=site_id,
                site_name=site_name,
            )

    event_table = (
        pd.concat(all_events, ignore_index=True)
        if all_events
        else pd.DataFrame()
    )

    summary_table = pd.DataFrame(summary_rows)

    output_tag = (
        "all_gauges"
        if args.site_id.lower() == "all"
        else args.site_id.zfill(8)
    )

    event_file = OUTPUT_DIR / f"{output_tag}_flood_events.csv"
    summary_file = (
        OUTPUT_DIR
        / f"{output_tag}_flood_duration_summary.csv"
    )

    event_table.to_csv(event_file, index=False)
    summary_table.to_csv(summary_file, index=False)

    print("\nFlood-duration summary:")
    print(summary_table.to_string(index=False))

    print("\nOutputs:")
    print(f"  Events:  {event_file}")
    print(f"  Summary: {summary_file}")
    print(f"  Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()