#!/usr/bin/env python3
# coding: utf-8

"""
Compute daily flow-duration curves for all USGS gauges.

The script:

1. Loads the original long-format USGS time-series file.
2. Selects streamflow observations only.
3. Computes daily mean discharge using a minimum completeness threshold.
4. Normalizes discharge by drainage area.
5. Computes one FDC per gauge.
6. Creates:
   - individual raw-discharge FDCs
   - one combined raw-discharge FDC
   - one combined normalized-discharge FDC
7. Exports daily flows, FDC values, and selected flow quantiles.

Run from the repository root:

    # Analyze the default target gauge

    python scripts/05_compute_flow_duration_curves.py

    # Analyze a selected gauge

    python scripts/05_compute_flow_duration_curves.py \

        --site-id 08210000

    # Analyze all gauges

    python scripts/05_compute_flow_duration_curves.py \

        --site-id all
s
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MIN_DAILY_COMPLETENESS = 0.80
EXPECTED_OBSERVATIONS_PER_DAY = 96  # nominal 15-minute USGS observations

MIN_DAILY_OBSERVATIONS = int(
    np.ceil(
        MIN_DAILY_COMPLETENESS
        * EXPECTED_OBSERVATIONS_PER_DAY
    )
)

FDC_PROBABILITIES = [5, 10, 25, 50, 75, 90, 95]


# ---------------------------------------------------------------------
# Paths
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
    / "flow_duration_curves"
)

FIGURE_DIR = (
    ROOT
    / "docs"
    / "figures"
    / "flow_duration_curves"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def load_streamflow(path: Path) -> pd.DataFrame:
    """Load original long-format data and retain streamflow observations."""

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(
        path,
        parse_dates=["datetime"],
        dtype={"site_id": "string"},
        low_memory=False,
    )

    required_columns = {
        "site_id",
        "datetime",
        "parameter_name",
        "value",
        "site_name",
        "river",
        "drainage_area_sqmi",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Input file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df["site_id"] = df["site_id"].str.zfill(8)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["drainage_area_sqmi"] = pd.to_numeric(
        df["drainage_area_sqmi"],
        errors="coerce",
    )

    flow = (
        df.loc[
            df["parameter_name"].eq("streamflow_cfs"),
            [
                "site_id",
                "site_name",
                "river",
                "datetime",
                "value",
                "drainage_area_sqmi",
            ],
        ]
        .rename(columns={"value": "streamflow_cfs"})
        .copy()
    )

    flow = flow.drop_duplicates(
        subset=["site_id", "datetime"],
        keep="last",
    )

    flow = flow.sort_values(["site_id", "datetime"])

    return flow


def compute_daily_flow(flow: pd.DataFrame, min_daily_observations: int,) -> pd.DataFrame:
    """Compute daily mean discharge with a completeness requirement."""

    daily = (
        flow.set_index("datetime")
        .groupby("site_id")["streamflow_cfs"]
        .resample("1D")
        .agg(
            daily_mean_cfs="mean",
            valid_subdaily_observations="count",
        )
        .reset_index()
    )

    daily["daily_complete"] = (
        daily["valid_subdaily_observations"]
        >= min_daily_observations
    )

    daily.loc[
        ~daily["daily_complete"],
        "daily_mean_cfs",
    ] = np.nan

    metadata = (
        flow[
            [
                "site_id",
                "site_name",
                "river",
                "drainage_area_sqmi",
            ]
        ]
        .drop_duplicates("site_id")
    )

    daily = daily.merge(
        metadata,
        on="site_id",
        how="left",
        validate="many_to_one",
    )

    daily["specific_discharge_cfs_sqmi"] = (
        daily["daily_mean_cfs"]
        / daily["drainage_area_sqmi"]
    )

    # 1 cfs/mi² is approximately 0.9826 mm/day.
    daily["runoff_mm_day"] = (
        daily["specific_discharge_cfs_sqmi"]
        * 0.9826
    )

    return daily


def compute_fdc(
    values: pd.Series,
    value_name: str,
) -> pd.DataFrame:
    """Compute an empirical exceedance-probability duration curve."""

    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    clean = clean[clean >= 0]

    if clean.empty:
        return pd.DataFrame(
            columns=["exceedance_percent", value_name]
        )

    sorted_values = np.sort(clean.to_numpy())[::-1]

    ranks = np.arange(
        1,
        len(sorted_values) + 1,
    )

    exceedance_percent = (
        100
        * ranks
        / (len(sorted_values) + 1)
    )

    return pd.DataFrame(
        {
            "exceedance_percent": exceedance_percent,
            value_name: sorted_values,
        }
    )


def discharge_at_exceedance(
    values: pd.Series,
    probabilities: list[int],
) -> dict[str, float]:
    """
    Return discharge exceeded for each requested percentage.

    Q90 means the discharge equaled or exceeded 90% of the time.
    """

    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    clean = clean[clean >= 0]

    if clean.empty:
        return {
            f"Q{probability}": np.nan
            for probability in probabilities
        }

    return {
        f"Q{probability}": float(
            np.percentile(
                clean,
                100 - probability,
            )
        )
        for probability in probabilities
    }


def safe_filename(text: str) -> str:
    """Convert text into a simple filename-friendly string."""

    cleaned = "".join(
        character if character.isalnum() else "_"
        for character in text
    )

    return "_".join(
        part
        for part in cleaned.split("_")
        if part
    )


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

def build_fdc_tables(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create long-format FDC and flow-quantile tables."""

    fdc_frames: list[pd.DataFrame] = []
    quantile_rows: list[dict[str, object]] = []

    for site_id, group in daily.groupby("site_id"):
        site_name = group["site_name"].iloc[0]
        river = group["river"].iloc[0]
        drainage_area = group["drainage_area_sqmi"].iloc[0]

        raw_fdc = compute_fdc(
            group["daily_mean_cfs"],
            value_name="daily_mean_cfs",
        )

        normalized_fdc = compute_fdc(
            group["specific_discharge_cfs_sqmi"],
            value_name="specific_discharge_cfs_sqmi",
        )

        runoff_fdc = compute_fdc(
            group["runoff_mm_day"],
            value_name="runoff_mm_day",
        )

        gauge_fdc = raw_fdc.merge(
            normalized_fdc,
            on="exceedance_percent",
            how="outer",
        ).merge(
            runoff_fdc,
            on="exceedance_percent",
            how="outer",
        )

        gauge_fdc.insert(0, "site_id", site_id)
        gauge_fdc.insert(1, "site_name", site_name)
        gauge_fdc.insert(2, "river", river)
        gauge_fdc.insert(
            3,
            "drainage_area_sqmi",
            drainage_area,
        )

        fdc_frames.append(gauge_fdc)

        raw_quantiles = discharge_at_exceedance(
            group["daily_mean_cfs"],
            FDC_PROBABILITIES,
        )

        normalized_quantiles = discharge_at_exceedance(
            group["specific_discharge_cfs_sqmi"],
            FDC_PROBABILITIES,
        )

        quantile_row: dict[str, object] = {
            "site_id": site_id,
            "site_name": site_name,
            "river": river,
            "drainage_area_sqmi": drainage_area,
            "valid_daily_values": int(
                group["daily_mean_cfs"].notna().sum()
            ),
            "incomplete_days": int(
                (~group["daily_complete"]).sum()
            ),
        }

        for name, value in raw_quantiles.items():
            quantile_row[f"{name}_cfs"] = value

        for name, value in normalized_quantiles.items():
            quantile_row[
                f"{name}_cfs_per_sqmi"
            ] = value

        quantile_rows.append(quantile_row)

    fdc_table = pd.concat(
        fdc_frames,
        ignore_index=True,
    )

    quantile_table = pd.DataFrame(quantile_rows)

    return fdc_table, quantile_table


def plot_individual_fdcs(
    fdc_table: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    """Create one raw-discharge FDC figure per gauge."""

    for site_id, group in fdc_table.groupby("site_id"):
        site_name = group["site_name"].iloc[0]
        river = group["river"].iloc[0]
        drainage_area = group["drainage_area_sqmi"].iloc[0]

        daily_valid = daily.loc[

            (daily["site_id"] == site_id)

            & (daily["daily_mean_cfs"].notna())

        ].copy()

        start_date = daily_valid["datetime"].min()

        end_date = daily_valid["datetime"].max()

        study_period = (

            f"{start_date:%Y-%m-%d} – {end_date:%Y-%m-%d}"

        )

        n_days = len(daily_valid)

        plot_data = group.dropna(
            subset=["daily_mean_cfs"]
        )

        figure, axis = plt.subplots(
            figsize=(8, 5)
        )

        axis.plot(
            plot_data["exceedance_percent"],
            plot_data["daily_mean_cfs"],
            linewidth=1.8,
        )

        axis.set_yscale("log")
        axis.set_xlim(0, 100)

        axis.set_xlabel(
            "Exceedance probability (%)"
        )
        axis.set_ylabel(
            "Daily mean discharge (ft³/s)"
        )

        axis.set_title(
            f"{site_name}\n"
            f"USGS {site_id} | {river} | {drainage_area:,.0f} mi²\n"
            f"{study_period} ({n_days:,} valid daily means)"
        )

        axis.grid(
            True,
            which="both",
            alpha=0.3,
        )

        figure.tight_layout()

        output_name = (
            f"{site_id}_"
            f"{safe_filename(site_name)}_fdc.png"
        )

        figure.savefig(
            FIGURE_DIR / output_name,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(figure)


def plot_combined_raw_fdc(
    fdc_table: pd.DataFrame,
) -> None:
    """Overlay raw-discharge FDCs for all gauges."""

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    for site_id, group in fdc_table.groupby("site_id"):
        site_name = group["site_name"].iloc[0]

        plot_data = group.dropna(
            subset=["daily_mean_cfs"]
        )

        axis.plot(
            plot_data["exceedance_percent"],
            plot_data["daily_mean_cfs"],
            linewidth=1.5,
            label=f"{site_id} — {site_name}",
        )

    axis.set_yscale("log")
    axis.set_xlim(0, 100)

    axis.set_xlabel(
        "Exceedance probability (%)"
    )
    axis.set_ylabel(
        "Daily mean discharge (ft³/s)"
    )
    axis.set_title(
        "Daily Flow-Duration Curves"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend(
        fontsize=8,
        loc="best",
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIR / "all_gauges_fdc_raw_discharge.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_combined_normalized_fdc(
    fdc_table: pd.DataFrame,
) -> None:
    """Overlay drainage-area-normalized FDCs for all gauges."""

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    for site_id, group in fdc_table.groupby("site_id"):
        site_name = group["site_name"].iloc[0]

        plot_data = group.dropna(
            subset=["specific_discharge_cfs_sqmi"]
        )

        axis.plot(
            plot_data["exceedance_percent"],
            plot_data["specific_discharge_cfs_sqmi"],
            linewidth=1.5,
            label=f"{site_id} — {site_name}",
        )

    axis.set_yscale("log")
    axis.set_xlim(0, 100)

    axis.set_xlabel(
        "Exceedance probability (%)"
    )
    axis.set_ylabel(
        "Specific discharge (ft³/s/mi²)"
    )

    axis.set_title(
        "Drainage-Area-Normalized Flow-Duration Curves"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend(
        fontsize=8,
        loc="best",
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIR / "all_gauges_fdc_specific_discharge.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_combined_runoff_fdc(
    fdc_table: pd.DataFrame,
) -> None:
    """Overlay FDCs expressed as runoff depth in mm/day."""

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    for site_id, group in fdc_table.groupby("site_id"):
        site_name = group["site_name"].iloc[0]

        plot_data = group.dropna(
            subset=["runoff_mm_day"]
        )

        axis.plot(
            plot_data["exceedance_percent"],
            plot_data["runoff_mm_day"],
            linewidth=1.5,
            label=f"{site_id} — {site_name}",
        )

    axis.set_yscale("log")
    axis.set_xlim(0, 100)

    axis.set_xlabel(
        "Exceedance probability (%)"
    )
    axis.set_ylabel(
        "Equivalent runoff depth (mm/day)"
    )

    axis.set_title(
        "Flow-Duration Curves as Runoff Depth"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    axis.legend(
        fontsize=8,
        loc="best",
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_DIR / "all_gauges_fdc_runoff_mm_day.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

def select_gauges(
    flow: pd.DataFrame,
    site_id: str,
) -> pd.DataFrame:
    """Select one gauge or retain all available gauges."""

    requested_site = str(site_id).strip()

    if requested_site.lower() == "all":
        return flow.copy()

    requested_site = requested_site.zfill(8)

    available_sites = sorted(
        flow["site_id"].dropna().unique().tolist()
    )

    if requested_site not in available_sites:
        available_text = ", ".join(available_sites)

        raise ValueError(
            f"USGS site {requested_site} was not found.\n"
            f"Available sites: {available_text}"
        )

    selected = flow.loc[
        flow["site_id"].eq(requested_site)
    ].copy()

    return selected

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute daily flow-duration curves from original "
            "USGS streamflow observations."
        )
    )

    parser.add_argument(
        "--site-id",
        type=str,
        default="08210000",
        help=(
            "Eight-digit USGS site ID to analyze. "
            "Use 'all' to process every available gauge. "
            "Default: 08210000"
        ),
    )

    parser.add_argument(
        "--min-daily-completeness",
        type=float,
        default=0.80,
        help=(
            "Minimum fraction of expected subdaily observations "
            "required to retain a daily mean. Default: 0.80"
        ),
    )

    return parser.parse_args()
# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not 0 < args.min_daily_completeness <= 1:
        raise ValueError(
            "--min-daily-completeness must be greater than 0 "
            "and no greater than 1."
        )

    min_daily_observations = int(
        np.ceil(
            args.min_daily_completeness
            * EXPECTED_OBSERVATIONS_PER_DAY
        )
    )

    print(f"Reading: {INPUT_FILE}")

    flow = load_streamflow(INPUT_FILE)

    print(
        f"Loaded {len(flow):,} streamflow observations "
        f"for {flow['site_id'].nunique()} gauges."
    )

    flow = select_gauges(
        flow=flow,
        site_id=args.site_id,
    )

    selected_sites = sorted(
        flow["site_id"].unique().tolist()
    )

    print(
        "Selected gauge(s): "
        + ", ".join(selected_sites)
    )

    gauge_information = (
        flow[
            [
                "site_id",
                "site_name",
                "river",
                "drainage_area_sqmi",
            ]
        ]
        .drop_duplicates("site_id")
    )

    print("\nGauge information:")
    print(
        gauge_information.to_string(
            index=False
        )
    )

    print(
        "\nDaily completeness requirement: "
        f"{args.min_daily_completeness:.0%}"
    )

    print(
        "Minimum valid subdaily observations per day: "
        f"{min_daily_observations} of "
        f"{EXPECTED_OBSERVATIONS_PER_DAY}"
    )

    daily = compute_daily_flow(
        flow=flow,
        min_daily_observations=min_daily_observations,
    )

    fdc_table, quantile_table = build_fdc_tables(
        daily
    )

    if args.site_id.lower() == "all":
        output_tag = "all_gauges"
    else:
        output_tag = selected_sites[0]

    daily_output = (
        OUTPUT_DIR
        / f"{output_tag}_daily_mean_streamflow.csv"
    )

    fdc_output = (
        OUTPUT_DIR
        / f"{output_tag}_flow_duration_curve.csv"
    )

    quantile_output = (
        OUTPUT_DIR
        / f"{output_tag}_flow_quantiles.csv"
    )

    daily.to_csv(
        daily_output,
        index=False,
    )

    fdc_table.to_csv(
        fdc_output,
        index=False,
    )

    quantile_table.to_csv(
        quantile_output,
        index=False,
    )

    plot_individual_fdcs(fdc_table=fdc_table,
                         daily=daily)

    if len(selected_sites) > 1:
        plot_combined_raw_fdc(fdc_table)
        plot_combined_normalized_fdc(fdc_table)
        plot_combined_runoff_fdc(fdc_table)

    print("\nFlow-duration statistics:")
    print(
        quantile_table.to_string(
            index=False
        )
    )

    print("\nOutputs created:")
    print(f"  Daily flow:      {daily_output}")
    print(f"  FDC table:       {fdc_output}")
    print(f"  Flow quantiles:  {quantile_output}")
    print(f"  Figures:         {FIGURE_DIR}")


if __name__ == "__main__":
    main()
