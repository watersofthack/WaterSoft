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

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "flood_wave_propagation"
)

FIGURE_DIR = (
    ROOT
    / "docs"
    / "figures"
    / "flood_wave_propagation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


TARGET_SITE_ID = "08210000"

# Selected upstream main-stem and tributary gauges that are

# hydrologically relevant to the target gauge during the analyzed events.

# The target gauge is placed last in the subplot order.

NETWORK_SITE_IDS = [
    "08194000",
    "08194500",
    "08206600",
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
    "action": "#D6D600",      # yellow
    "minor": "#FFA500",       # orange
    "moderate": "#FF0000",    # red
    "major": "#C000C0",       # purple/magenta
}

THRESHOLD_LINE_STYLES = {
    "action": "-",
    "minor": "-",
    "moderate": "-",
    "major": "-",
}
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
        default="15min",
        help=(
            "Plotting resolution. Use '15min' for the original "
            "observations or 'hourly' for hourly means. Default: 15min"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_network_data(path: Path) -> pd.DataFrame:
    """Load streamflow, stage, metadata, and flood thresholds."""

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

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
        parse_dates=["datetime"],
        dtype={"site_id": "string"},
        low_memory=False,
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

    available_sites = set(
        df["site_id"].dropna().unique()
    )

    missing_sites = [
        site_id
        for site_id in NETWORK_SITE_IDS
        if site_id not in available_sites
    ]

    if missing_sites:
        raise ValueError(
            "The following requested gauges were not found: "
            + ", ".join(missing_sites)
        )

    df = df.loc[
        df["site_id"].isin(NETWORK_SITE_IDS)
    ].copy()

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
        stage.dropna(
            subset=["datetime", "gage_height_ft"]
        )
        .sort_values("datetime")
        .copy()
    )

    if data.empty or pd.isna(threshold_ft):
        return pd.DataFrame()

    data["above_threshold"] = (
        data["gage_height_ft"] >= threshold_ft
    )

    time_gap_minutes = (
        data["datetime"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    previous_above = data[
        "above_threshold"
    ].shift(fill_value=False)

    data["new_event"] = (
        data["above_threshold"]
        & (
            ~previous_above
            | time_gap_minutes.gt(MAX_GAP_MINUTES)
        )
    )

    data["event_number"] = (
        data["new_event"].cumsum()
    )

    event_data = data.loc[
        data["above_threshold"]
    ].copy()

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
        events["last_observation_time"]
        + nominal_interval
    )

    events["duration_hours"] = (
        events["end_time"]
        - events["start_time"]
    ).dt.total_seconds() / 3600

    events["duration_days"] = (
        events["duration_hours"] / 24
    )

    events["threshold_ft"] = threshold_ft

    events = (
        events.sort_values(
            "duration_hours",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    events["duration_rank"] = (
        np.arange(1, len(events) + 1)
    )

    return events


def get_ranked_major_events(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Identify and rank major events at the target gauge."""

    target = df.loc[
        df["site_id"].eq(TARGET_SITE_ID)
    ].copy()

    stage = (
        target.loc[
            target["parameter_name"].eq(
                "gage_height_ft"
            ),
            [
                "datetime",
                "value",
                "major_flood_stage_ft",
            ],
        ]
        .rename(
            columns={"value": "gage_height_ft"}
        )
        .copy()
    )

    threshold_values = stage[
        "major_flood_stage_ft"
    ].dropna()

    if threshold_values.empty:
        raise ValueError(
            f"No major-flood threshold is available "
            f"for target gauge {TARGET_SITE_ID}."
        )

    major_threshold_ft = float(
        threshold_values.iloc[0]
    )

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
        raise ValueError(
            "--event-rank must be at least 1."
        )

    available_ranks = events[
        "duration_rank"
    ].astype(int).tolist()

    if event_rank not in available_ranks:
        raise ValueError(
            f"Major event rank {event_rank} is not available. "
            f"Available ranks: {available_ranks}"
        )

    selected = events.loc[
        events["duration_rank"].eq(event_rank)
    ].iloc[0]

    return selected


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

    window_start = (
        selected_event["start_time"]
        - pd.Timedelta(days=days_before)
    )

    window_end = (
        selected_event["end_time"]
        + pd.Timedelta(days=days_after)
    )

    event_window = df.loc[
        df["datetime"].between(
            window_start,
            window_end,
            inclusive="both",
        )
    ].copy()

    if event_window.empty:
        raise ValueError(
            "No observations were found within the selected "
            "event window."
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
        group = event_window.loc[
            event_window["site_id"].eq(site_id)
        ]

        if group.empty:
            continue

        site_series[site_id] = (
            create_site_timeseries(
                group=group,
                resample=resample,
            )
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

        flow = series[
            "streamflow_cfs"
        ].dropna()

        stage = series[
            "gage_height_ft"
        ].dropna()

        flow_peak_time = (
            flow.idxmax()
            if not flow.empty
            else pd.NaT
        )

        stage_peak_time = (
            stage.idxmax()
            if not stage.empty
            else pd.NaT
        )

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
                    float(flow.max())
                    if not flow.empty
                    else np.nan
                ),
                "streamflow_peak_time": (
                    flow_peak_time
                ),
                "peak_stage_ft": (
                    float(stage.max())
                    if not stage.empty
                    else np.nan
                ),
                "stage_peak_time": (
                    stage_peak_time
                ),
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
        target_flow_time
        - peaks["streamflow_peak_time"]
    ).dt.total_seconds() / 3600

    peaks["stage_peak_lag_to_target_hours"] = (
        target_stage_time
        - peaks["stage_peak_time"]
    ).dt.total_seconds() / 3600

    # Positive lag means the upstream/tributary gauge peaked before
    # the target gauge.
    peaks["network_order"] = peaks[
        "site_id"
    ].map(
        {
            site_id: index
            for index, site_id
            in enumerate(NETWORK_SITE_IDS)
        }
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

    metadata = event_window.loc[
        event_window["site_id"].eq(site_id)
    ]

    if metadata.empty:
        raise ValueError(
            f"No metadata found for site {site_id}."
        )

    return metadata.iloc[0]


def format_time_axis(axis: plt.Axes) -> None:
    """Apply readable datetime formatting."""

    locator = mdates.AutoDateLocator(
        minticks=5,
        maxticks=10,
    )

    formatter = mdates.ConciseDateFormatter(
        locator
    )

    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)


def plot_streamflow_propagation(
    site_series: dict[str, pd.DataFrame],
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
        site_id
        for site_id in NETWORK_SITE_IDS
        if site_id in site_series
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

        flow = series[
            "streamflow_cfs"
        ].dropna()

        axis.plot(
            flow.index,
            flow.values,
            linewidth=1.3,
        )

        peak_row = peaks.loc[
            peaks["site_id"].eq(site_id)
        ]

        if not peak_row.empty:
            peak_time = peak_row[
                "streamflow_peak_time"
            ].iloc[0]

            peak_value = peak_row[
                "peak_streamflow_cfs"
            ].iloc[0]

            lag_hours = peak_row[
                "flow_peak_lag_to_target_hours"
            ].iloc[0]

            if pd.notna(peak_time):
                axis.axvline(
                    peak_time,
                    linestyle="--",
                    linewidth=1,
                )

                label_text = (
                    f"Peak: {peak_value:,.0f} cfs"
                )

                if (
                    site_id != TARGET_SITE_ID
                    and pd.notna(lag_hours)
                ):
                    label_text += (
                        f"\nLead to target: "
                        f"{lag_hours:.1f} h"
                    )

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
            f"{metadata['site_name']} | "
            f"USGS {site_id}",
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

    figure.tight_layout()

    output_path = (
        FIGURE_DIR
        / (
            f"{TARGET_SITE_ID}_major_event_rank_"
            f"{event_rank}_{resample}_"
            f"streamflow_propagation.png"
        )
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
        site_id
        for site_id in NETWORK_SITE_IDS
        if site_id in site_series
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

        stage = series[
            "gage_height_ft"
        ].dropna()

        axis.plot(
            stage.index,
            stage.values,
            linewidth=1.3,
            label="Gage height",
        )

        for threshold_name, column in (
            THRESHOLD_COLUMNS.items()
        ):
            threshold_ft = metadata[column]

            if pd.isna(threshold_ft):
                continue

            axis.axhline(
                threshold_ft,
                color=THRESHOLD_COLORS[threshold_name],
                linestyle=THRESHOLD_LINE_STYLES[threshold_name],
                linewidth=1.8,
                label=threshold_name.title(),
            )

        peak_row = peaks.loc[
            peaks["site_id"].eq(site_id)
        ]

        if not peak_row.empty:
            peak_time = peak_row[
                "stage_peak_time"
            ].iloc[0]

            peak_value = peak_row[
                "peak_stage_ft"
            ].iloc[0]

            lag_hours = peak_row[
                "stage_peak_lag_to_target_hours"
            ].iloc[0]

            if pd.notna(peak_time):
                axis.axvline(
                    peak_time,
                    linestyle="--",
                    linewidth=1,
                )

                label_text = (
                    f"Peak: {peak_value:.2f} ft"
                )

                if (
                    site_id != TARGET_SITE_ID
                    and pd.notna(lag_hours)
                ):
                    label_text += (
                        f"\nLead to target: "
                        f"{lag_hours:.1f} h"
                    )

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
            f"{metadata['site_name']} | "
            f"USGS {site_id}",
            loc="left",
            fontsize=12,
        )

        axis.grid(
            True,
            alpha=0.25,
        )

        axis.tick_params(axis="both", labelsize=12)

        axis.legend(
            fontsize=8,
            loc="upper left",
            ncol=3,
        )

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

    figure.tight_layout()

    output_path = (
        FIGURE_DIR
        / (
            f"{TARGET_SITE_ID}_major_event_rank_"
            f"{event_rank}_{resample}_"
            f"stage_propagation.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.days_before < 0:
        raise ValueError(
            "--days-before cannot be negative."
        )

    if args.days_after < 0:
        raise ValueError(
            "--days-after cannot be negative."
        )

    print(f"Reading: {INPUT_FILE}")

    df = load_network_data(INPUT_FILE)

    major_events = get_ranked_major_events(df)

    print(
        "\nMajor-stage events at target gauge "
        f"{TARGET_SITE_ID}, ranked longest to shortest:"
    )

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

    print(
        f"\nSelected major event rank: "
        f"{args.event_rank}"
    )

    print(
        f"Start:    "
        f"{selected_event['start_time']}"
    )

    print(
        f"End:      "
        f"{selected_event['end_time']}"
    )

    print(
        f"Duration: "
        f"{selected_event['duration_hours']:.2f} hours "
        f"({selected_event['duration_days']:.2f} days)"
    )

    print(
        f"Peak stage at target: "
        f"{selected_event['peak_stage_ft']:.2f} ft"
    )

    event_window, window_start, window_end = (
        extract_event_window(
            df=df,
            selected_event=selected_event,
            days_before=args.days_before,
            days_after=args.days_after,
        )
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
        f"{TARGET_SITE_ID}_major_event_"
        f"rank_{args.event_rank}_{args.resample}"
    )

    ranked_events_file = (
        OUTPUT_DIR
        / f"{TARGET_SITE_ID}_ranked_major_events.csv"
    )

    selected_event_file = (
        OUTPUT_DIR
        / f"{output_tag}_selected_event.csv"
    )

    event_window_file = (
        OUTPUT_DIR
        / f"{output_tag}_network_timeseries.csv"
    )

    peak_summary_file = (
        OUTPUT_DIR
        / f"{output_tag}_peak_summary.csv"
    )

    major_events.to_csv(
        ranked_events_file,
        index=False,
    )

    pd.DataFrame(
        [selected_event.to_dict()]
    ).to_csv(
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

    streamflow_figure = (
        plot_streamflow_propagation(
            site_series=site_series,
            event_window=event_window,
            peaks=peaks,
            selected_event=selected_event,
            event_rank=args.event_rank,
            window_start=window_start,
            window_end=window_end,
            resample=args.resample,
        )
    )

    stage_figure = plot_stage_propagation(
        site_series=site_series,
        event_window=event_window,
        peaks=peaks,
        selected_event=selected_event,
        event_rank=args.event_rank,
        window_start=window_start,
        window_end=window_end,
        resample=args.resample,
    )

    print("\nOutputs created:")
    print(
        f"  Ranked major events: {ranked_events_file}"
    )
    print(
        f"  Selected event:      {selected_event_file}"
    )
    print(
        f"  Network time series: {event_window_file}"
    )
    print(
        f"  Peak summary:        {peak_summary_file}"
    )
    print(
        f"  Streamflow figure:   {streamflow_figure}"
    )
    print(
        f"  Stage figure:        {stage_figure}"
    )


if __name__ == "__main__":
    main()