#!/usr/bin/env python3
# coding: utf-8

"""
Fill rare gaps in downloaded USGS streamflow and gage-height time series.

The script automatically locates the project root, reads the downloaded
wide-format USGS CSV, fills missing values independently for each gauge
using the nearest valid time step, and saves new filled files.

The original files are not overwritten.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


VALUE_COLUMNS = [
    "streamflow_cfs",
    "gage_height_ft",
]


def find_project_root() -> Path:
    """
    Find the repository root by searching the current directory,
    parent directories, and the script location for a data/ directory.
    """

    candidate_paths = []

    # Location from which the script was executed.
    current_dir = Path.cwd().resolve()
    candidate_paths.extend([current_dir, *current_dir.parents])

    # Physical location of this Python script.
    script_dir = Path(__file__).resolve().parent
    candidate_paths.extend([script_dir, *script_dir.parents])

    checked_paths = set()

    for candidate in candidate_paths:
        if candidate in checked_paths:
            continue

        checked_paths.add(candidate)

        if (candidate / "data").is_dir():
            return candidate

    checked_text = "\n".join(
        f"  - {path}" for path in checked_paths
    )

    raise FileNotFoundError(
        "Could not locate the project root.\n"
        "Expected to find a directory containing data/.\n\n"
        f"Checked:\n{checked_text}"
    )


def detect_datetime_column(df: pd.DataFrame) -> str:
    """Identify the datetime column."""

    candidates = [
        "datetime",
        "date_time",
        "timestamp",
        "time",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not identify the datetime column. "
        f"Available columns: {list(df.columns)}"
    )


def fill_site_gaps(
    site_df: pd.DataFrame,
    datetime_column: str,
    value_columns: list[str],
) -> pd.DataFrame:
    """
    Fill gaps for one gauge using the nearest valid time step.

    Each gauge is processed independently, so observations from one
    station cannot be used to fill another station.
    """

    site_df = site_df.sort_values(datetime_column).copy()
    site_df = site_df.set_index(datetime_column)

    for column in value_columns:
        site_df[column] = site_df[column].interpolate(
            method="nearest"
        )

    return site_df.reset_index()


def create_missing_summary(
    df: pd.DataFrame,
    value_columns: list[str],
    suffix: str,
) -> pd.DataFrame:
    """Calculate missing counts and percentages by gauge."""

    summaries = []

    for column in value_columns:
        summary = (
            df.groupby("site_id")[column]
            .agg(
                observation_count="size",
                missing_count=lambda values: values.isna().sum(),
            )
            .reset_index()
        )

        summary["missing_percent"] = (
            100.0
            * summary["missing_count"]
            / summary["observation_count"]
        )

        summary["parameter_name"] = column

        summary = summary.rename(
            columns={
                "missing_count": f"missing_count_{suffix}",
                "missing_percent": f"missing_percent_{suffix}",
            }
        )

        summaries.append(summary)

    return pd.concat(summaries, ignore_index=True)


def wide_to_long(
    df: pd.DataFrame,
    datetime_column: str,
    value_columns: list[str],
) -> pd.DataFrame:
    """Convert the filled wide-format data to long format."""

    identifier_columns = [
        column
        for column in df.columns
        if column not in value_columns
    ]

    long_df = df.melt(
        id_vars=identifier_columns,
        value_vars=value_columns,
        var_name="parameter_name",
        value_name="value",
    )

    return long_df.sort_values(
        ["site_id", "parameter_name", datetime_column]
    ).reset_index(drop=True)


def main() -> None:
    project_root = find_project_root()

    timeseries_dir = (
        project_root
        / "data"
        / "raw"
        / "usgs"
        / "timeseries"
    )

    input_path = (
        timeseries_dir
        / "usgs_gauge_height_streamflow_wide.csv"
    )

    wide_output_path = (
        timeseries_dir
        / "usgs_gauge_height_streamflow_wide_filled.csv"
    )

    long_output_path = (
        timeseries_dir
        / "usgs_gauge_height_streamflow_long_filled.csv"
    )

    summary_output_path = (
        timeseries_dir
        / "usgs_gap_filling_summary.csv"
    )

    print(f"Project root:\n{project_root}")
    print(f"\nReading input file:\n{input_path}")

    if not input_path.exists():
        raise FileNotFoundError(
            "Could not find the downloaded wide-format file:\n"
            f"{input_path}\n\n"
            "Run 03_download_usgs_timeseries.py first."
        )

    df = pd.read_csv(
        input_path,
        dtype={"site_id": "string"},
        low_memory=False,
    )

    if "site_id" not in df.columns:
        raise ValueError(
            "The input file does not contain a site_id column."
        )

    datetime_column = detect_datetime_column(df)

    df[datetime_column] = pd.to_datetime(
        df[datetime_column],
        utc=True,
        errors="coerce",
    )

    invalid_datetime_count = int(
        df[datetime_column].isna().sum()
    )

    if invalid_datetime_count > 0:
        raise ValueError(
            f"Found {invalid_datetime_count:,} rows with "
            "invalid datetime values."
        )

    available_value_columns = [
        column
        for column in VALUE_COLUMNS
        if column in df.columns
    ]

    if not available_value_columns:
        raise ValueError(
            "The input file does not contain streamflow_cfs "
            "or gage_height_ft."
        )

    for column in available_value_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.sort_values(
        ["site_id", datetime_column]
    ).reset_index(drop=True)

    before_summary = create_missing_summary(
        df=df,
        value_columns=available_value_columns,
        suffix="before",
    )

    print("\nMissing values before filling:")

    for column in available_value_columns:
        missing_count = int(df[column].isna().sum())
        print(f"  {column}: {missing_count:,}")

    filled_parts = []

    for site_id, site_df in df.groupby(
        "site_id",
        sort=False,
        dropna=False,
    ):
        filled_site = fill_site_gaps(
            site_df=site_df,
            datetime_column=datetime_column,
            value_columns=available_value_columns,
        )

        filled_parts.append(filled_site)

    filled_df = pd.concat(
        filled_parts,
        ignore_index=True,
    )

    filled_df = filled_df.sort_values(
        ["site_id", datetime_column]
    ).reset_index(drop=True)

    after_summary = create_missing_summary(
        df=filled_df,
        value_columns=available_value_columns,
        suffix="after",
    )

    print("\nMissing values after filling:")

    for column in available_value_columns:
        missing_count = int(
            filled_df[column].isna().sum()
        )
        print(f"  {column}: {missing_count:,}")

    summary = before_summary.merge(
        after_summary[
            [
                "site_id",
                "parameter_name",
                "missing_count_after",
                "missing_percent_after",
            ]
        ],
        on=["site_id", "parameter_name"],
        how="left",
    )

    summary["filled_count"] = (
        summary["missing_count_before"]
        - summary["missing_count_after"]
    )

    summary = summary[
        [
            "site_id",
            "parameter_name",
            "observation_count",
            "missing_count_before",
            "missing_percent_before",
            "filled_count",
            "missing_count_after",
            "missing_percent_after",
        ]
    ].sort_values(
        ["site_id", "parameter_name"]
    )

    filled_df.to_csv(
        wide_output_path,
        index=False,
    )

    filled_long_df = wide_to_long(
        df=filled_df,
        datetime_column=datetime_column,
        value_columns=available_value_columns,
    )

    filled_long_df.to_csv(
        long_output_path,
        index=False,
    )

    summary.to_csv(
        summary_output_path,
        index=False,
    )

    print("\nGap-filling summary:")
    print(summary.to_string(index=False))

    print("\nSaved files:")

    print(
        "\nFilled wide-format observations:"
        f"\n{wide_output_path}"
    )

    print(
        "\nFilled long-format observations:"
        f"\n{long_output_path}"
    )

    print(
        "\nGap-filling summary:"
        f"\n{summary_output_path}"
    )


if __name__ == "__main__":
    main()