#!/usr/bin/env python3
# coding: utf-8

"""
Download, clean, export, and plot a USGS stage-discharge rating curve.

Default target:

    USGS site: 08210000
    NWM ReachID: 3168766

The script:

1. Queries the USGS Water Data STAC API.
2. Finds available rating files for the selected monitoring location.
3. Prefers the expanded-stage rating file ("exsa").
4. Downloads and preserves the original USGS RDB file.
5. Parses gauge height and discharge.
6. Creates an interpolation-ready rating table.
7. Produces:
   - rating curve in arithmetic coordinates
   - rating curve with logarithmic discharge axis
8. Exports station metadata and rating limits.

Run from the repository root:

    python scripts/06_download_usgs_rating_curve.py

For another station:

    python scripts/06_download_usgs_rating_curve.py \
        --site-id 08210000

Optionally select another rating type:

    python scripts/06_download_usgs_rating_curve.py \
        --site-id 08210000 \
        --rating-type base

Requirements:

    pip install pandas numpy matplotlib requests
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_SITE_ID = "08210000"
DEFAULT_NWM_REACH_ID = 3168766

STAC_SEARCH_URL = (
    "https://api.waterdata.usgs.gov/stac/v0/search"
)

REQUEST_TIMEOUT_SECONDS = 60

RATING_TYPE_PRIORITY = [
    "exsa",
    "base",
]


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_OUTPUT_DIR = (
    ROOT
    / "data"
    / "raw"
    / "usgs"
    / "rating_curves"
)

PROCESSED_OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "rating_curves"
)

FIGURE_DIR = (
    ROOT
    / "docs"
    / "figures"
    / "rating_curves"
)

RAW_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Download and process a USGS stage-discharge "
            "rating curve."
        )
    )

    parser.add_argument(
        "--site-id",
        type=str,
        default=DEFAULT_SITE_ID,
        help=(
            "USGS monitoring-location number. "
            f"Default: {DEFAULT_SITE_ID}"
        ),
    )

    parser.add_argument(
        "--nwm-reach-id",
        type=int,
        default=DEFAULT_NWM_REACH_ID,
        help=(
            "Associated NWM ReachID written to output metadata. "
            f"Default: {DEFAULT_NWM_REACH_ID}"
        ),
    )

    parser.add_argument(
        "--rating-type",
        type=str,
        choices=["auto", "exsa", "base", "corr"],
        default="auto",
        help=(
            "USGS rating-file type. The default 'auto' "
            "prefers exsa, then base."
        ),
    )

    parser.add_argument(
        "--allow-negative-discharge",
        action="store_true",
        help=(
            "Retain negative discharge values, if present. "
            "Normally these are removed."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# STAC discovery
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


def query_rating_assets(
    site_id: str,
) -> list[dict[str, Any]]:
    """
    Query the USGS STAC API for rating files associated
    with one monitoring location.
    """

    monitoring_location_id = f"USGS-{site_id}"

    params = {
        "collection": "ratings",
        "filter": (
            "monitoring_location_id="
            f"'{monitoring_location_id}'"
        ),
        "limit": 20,
    }

    print(
        "Querying USGS rating catalog for "
        f"{monitoring_location_id}..."
    )

    response = requests.get(
        STAC_SEARCH_URL,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    payload = response.json()
    features = payload.get("features", [])

    assets: list[dict[str, Any]] = []

    for feature in features:
        properties = feature.get("properties", {})
        asset = feature.get("assets", {}).get("data", {})

        file_type = properties.get("file_type")
        asset_url = asset.get("href")

        if not file_type or not asset_url:
            continue

        assets.append(
            {
                "file_type": str(file_type).lower(),
                "url": asset_url,
                "item_id": feature.get("id"),
                "updated_datetime": properties.get(
                    "datetime"
                ),
                "description": asset.get("description"),
            }
        )

    if not assets:
        raise RuntimeError(
            "No USGS rating files were returned for "
            f"site {site_id}."
        )

    return assets


def select_rating_asset(
    assets: list[dict[str, Any]],
    requested_type: str,
) -> dict[str, Any]:
    """Choose the desired rating asset."""

    available_types = sorted(
        {
            asset["file_type"]
            for asset in assets
        }
    )

    print(
        "Available rating types: "
        + ", ".join(available_types)
    )

    if requested_type != "auto":
        matches = [
            asset
            for asset in assets
            if asset["file_type"] == requested_type
        ]

        if not matches:
            raise ValueError(
                f"Rating type {requested_type!r} is unavailable. "
                f"Available types: {available_types}"
            )

        return matches[0]

    for preferred_type in RATING_TYPE_PRIORITY:
        matches = [
            asset
            for asset in assets
            if asset["file_type"] == preferred_type
        ]

        if matches:
            return matches[0]

    raise RuntimeError(
        "Neither an expanded-stage nor base rating file "
        f"was available. Available types: {available_types}"
    )


# ---------------------------------------------------------------------
# Download and parsing
# ---------------------------------------------------------------------

def download_text(url: str) -> str:
    """Download a text asset."""

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    return response.text


def extract_comment_metadata(
    text: str,
) -> dict[str, str]:
    """Extract selected metadata from USGS RDB comment lines."""

    metadata: dict[str, str] = {}

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped.startswith("#"):
            continue

        content = stripped.lstrip("#").strip()

        if "=" in content:
            key, value = content.split("=", maxsplit=1)

            key = (
                key.strip()
                .lower()
                .replace(" ", "_")
            )

            metadata[key] = value.strip()

    return metadata


def read_usgs_rdb(text: str) -> pd.DataFrame:
    """
    Parse a USGS tab-delimited RDB file.

    USGS RDB files generally contain:

    - comment lines beginning with #
    - one column-name row
    - one field-width/type row
    - tab-delimited data rows
    """

    non_comment_lines = [
        line
        for line in text.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
    ]

    if len(non_comment_lines) < 3:
        raise ValueError(
            "The downloaded rating file does not contain "
            "enough non-comment rows to be parsed."
        )

    column_names = non_comment_lines[0].split("\t")

    type_definition = non_comment_lines[1].split("\t")

    if len(column_names) != len(type_definition):
        raise ValueError(
            "Unexpected USGS RDB structure: the column-name "
            "and field-definition rows have different lengths."
        )

    data_text = "\n".join(
        non_comment_lines[2:]
    )

    table = pd.read_csv(
        io.StringIO(data_text),
        sep="\t",
        names=column_names,
        dtype="string",
    )

    return table


def identify_rating_columns(
    table: pd.DataFrame,
) -> tuple[str, str]:
    """
    Identify gauge-height and discharge columns.

    Common expanded-stage columns include:

        INDEP
        DEP

    Common descriptive alternatives may contain:

        stage
        gage
        height
        discharge
        flow
    """

    columns = list(table.columns)

    lower_lookup = {
        column.lower().strip(): column
        for column in columns
    }

    exact_stage_candidates = [
        "indep",
        "gage_height",
        "gage_height_ft",
        "stage",
        "stage_ft",
    ]

    exact_discharge_candidates = [
        "dep",
        "discharge",
        "discharge_cfs",
        "flow",
        "streamflow",
    ]

    stage_column = next(
        (
            lower_lookup[candidate]
            for candidate in exact_stage_candidates
            if candidate in lower_lookup
        ),
        None,
    )

    discharge_column = next(
        (
            lower_lookup[candidate]
            for candidate in exact_discharge_candidates
            if candidate in lower_lookup
        ),
        None,
    )

    if stage_column is None:
        stage_matches = [
            column
            for column in columns
            if any(
                token in column.lower()
                for token in [
                    "gage",
                    "stage",
                    "height",
                    "indep",
                ]
            )
        ]

        if stage_matches:
            stage_column = stage_matches[0]

    if discharge_column is None:
        discharge_matches = [
            column
            for column in columns
            if any(
                token in column.lower()
                for token in [
                    "discharge",
                    "streamflow",
                    "flow",
                    "dep",
                ]
            )
        ]

        if discharge_matches:
            discharge_column = discharge_matches[0]

    if stage_column is None or discharge_column is None:
        raise ValueError(
            "Could not identify stage and discharge columns. "
            f"Columns found: {columns}"
        )

    if stage_column == discharge_column:
        raise ValueError(
            "The same RDB column was identified as both stage "
            "and discharge."
        )

    return stage_column, discharge_column


def clean_rating_table(
    raw_table: pd.DataFrame,
    site_id: str,
    nwm_reach_id: int,
    rating_type: str,
    allow_negative_discharge: bool,
) -> pd.DataFrame:
    """Create a clean interpolation-ready rating table."""

    stage_column, discharge_column = (
        identify_rating_columns(raw_table)
    )

    print(
        f"Using stage column: {stage_column}"
    )

    print(
        f"Using discharge column: {discharge_column}"
    )

    rating = pd.DataFrame(
        {
            "gauge_height_ft": pd.to_numeric(
                raw_table[stage_column],
                errors="coerce",
            ),
            "discharge_cfs": pd.to_numeric(
                raw_table[discharge_column],
                errors="coerce",
            ),
        }
    )

    rating = rating.dropna(
        subset=[
            "gauge_height_ft",
            "discharge_cfs",
        ]
    )

    if not allow_negative_discharge:
        rating = rating.loc[
            rating["discharge_cfs"] >= 0
        ].copy()

    rating = rating.drop_duplicates(
        subset=[
            "gauge_height_ft",
            "discharge_cfs",
        ]
    )

    rating = rating.sort_values(
        [
            "gauge_height_ft",
            "discharge_cfs",
        ]
    ).reset_index(drop=True)

    if rating.empty:
        raise ValueError(
            "No valid stage-discharge pairs remained after "
            "cleaning the rating table."
        )

    rating.insert(
        0,
        "site_id",
        site_id,
    )

    rating.insert(
        1,
        "nwm_reach_id",
        nwm_reach_id,
    )

    rating.insert(
        2,
        "rating_type",
        rating_type,
    )

    rating["discharge_cms"] = (
        rating["discharge_cfs"]
        / 35.3146667215
    )

    rating["gauge_height_m"] = (
        rating["gauge_height_ft"]
        * 0.3048
    )

    return rating


# ---------------------------------------------------------------------
# Rating validation
# ---------------------------------------------------------------------

def validate_rating_monotonicity(
    rating: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check whether discharge increases monotonically with stage.

    A direct inverse interpolation Q -> H requires discharge
    to be monotonic after sorting by gauge height.
    """

    checked = rating.copy()

    checked["discharge_change_cfs"] = (
        checked["discharge_cfs"].diff()
    )

    checked["nonincreasing_discharge"] = (
        checked["discharge_change_cfs"] <= 0
    )

    violations = checked.loc[
        checked["nonincreasing_discharge"].fillna(False)
    ]

    if violations.empty:
        print(
            "Rating monotonicity check: passed."
        )
    else:
        print(
            "Warning: "
            f"{len(violations)} non-increasing rating steps "
            "were detected."
        )

    return checked


def create_inverse_rating_table(
    rating: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a discharge-sorted table for Q -> gauge-height
    interpolation.

    Duplicate discharge values are combined using their median
    gauge height.
    """

    inverse = (
        rating.groupby(
            "discharge_cfs",
            as_index=False,
        )
        .agg(
            gauge_height_ft=(
                "gauge_height_ft",
                "median",
            ),
            discharge_cms=(
                "discharge_cms",
                "first",
            ),
            site_id=(
                "site_id",
                "first",
            ),
            nwm_reach_id=(
                "nwm_reach_id",
                "first",
            ),
            rating_type=(
                "rating_type",
                "first",
            ),
        )
        .sort_values("discharge_cfs")
        .reset_index(drop=True)
    )

    inverse["gauge_height_m"] = (
        inverse["gauge_height_ft"]
        * 0.3048
    )

    return inverse


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_rating_curve(
    rating: pd.DataFrame,
    site_id: str,
) -> None:
    """Plot stage-discharge rating in arithmetic coordinates."""

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.plot(
        rating["gauge_height_ft"],
        rating["discharge_cfs"],
        linewidth=2,
    )

    axis.scatter(
        rating["gauge_height_ft"],
        rating["discharge_cfs"],
        s=10,
        alpha=0.5,
    )

    axis.set_xlabel(
        "Gauge height (ft)"
    )

    axis.set_ylabel(
        "Discharge (ft³/s)"
    )

    axis.set_title(
        f"USGS {site_id} Stage–Discharge Rating Curve"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    output_path = (
        FIGURE_DIR
        / f"{site_id}_rating_curve_linear.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved: {output_path}"
    )


def plot_rating_curve_log_discharge(
    rating: pd.DataFrame,
    site_id: str,
) -> None:
    """Plot stage-discharge rating with logarithmic discharge."""

    plot_data = rating.loc[
        rating["discharge_cfs"] > 0
    ].copy()

    if plot_data.empty:
        print(
            "Skipping logarithmic rating plot because no "
            "positive discharge values are available."
        )

        return

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.plot(
        plot_data["gauge_height_ft"],
        plot_data["discharge_cfs"],
        linewidth=2,
    )

    axis.scatter(
        plot_data["gauge_height_ft"],
        plot_data["discharge_cfs"],
        s=10,
        alpha=0.5,
    )

    axis.set_yscale("log")

    axis.set_xlabel(
        "Gauge height (ft)"
    )

    axis.set_ylabel(
        "Discharge (ft³/s)"
    )

    axis.set_title(
        f"USGS {site_id} Stage–Discharge Rating Curve"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.3,
    )

    figure.tight_layout()

    output_path = (
        FIGURE_DIR
        / f"{site_id}_rating_curve_log_discharge.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved: {output_path}"
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def build_rating_summary(
    rating: pd.DataFrame,
    asset: dict[str, Any],
    site_id: str,
    nwm_reach_id: int,
) -> dict[str, Any]:
    """Create a rating-curve summary dictionary."""

    return {
        "site_id": site_id,
        "nwm_reach_id": nwm_reach_id,
        "rating_type": asset["file_type"],
        "rating_item_id": asset.get("item_id"),
        "rating_updated_datetime": asset.get(
            "updated_datetime"
        ),
        "rating_source_url": asset.get("url"),
        "number_of_rating_points": int(
            len(rating)
        ),
        "minimum_gauge_height_ft": float(
            rating["gauge_height_ft"].min()
        ),
        "maximum_gauge_height_ft": float(
            rating["gauge_height_ft"].max()
        ),
        "minimum_discharge_cfs": float(
            rating["discharge_cfs"].min()
        ),
        "maximum_discharge_cfs": float(
            rating["discharge_cfs"].max()
        ),
        "minimum_discharge_cms": float(
            rating["discharge_cms"].min()
        ),
        "maximum_discharge_cms": float(
            rating["discharge_cms"].max()
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    site_id = normalize_site_id(
        args.site_id
    )

    assets = query_rating_assets(
        site_id=site_id
    )

    selected_asset = select_rating_asset(
        assets=assets,
        requested_type=args.rating_type,
    )

    rating_type = selected_asset["file_type"]
    source_url = selected_asset["url"]

    print(
        f"Selected rating type: {rating_type}"
    )

    print(
        f"Downloading: {source_url}"
    )

    rating_text = download_text(
        source_url
    )

    raw_output_path = (
        RAW_OUTPUT_DIR
        / f"USGS.{site_id}.{rating_type}.rdb"
    )

    raw_output_path.write_text(
        rating_text,
        encoding="utf-8",
    )

    print(
        f"Saved original rating file: {raw_output_path}"
    )

    metadata = extract_comment_metadata(
        rating_text
    )

    raw_table = read_usgs_rdb(
        rating_text
    )

    print(
        "Raw rating columns: "
        + ", ".join(raw_table.columns)
    )

    rating = clean_rating_table(
        raw_table=raw_table,
        site_id=site_id,
        nwm_reach_id=args.nwm_reach_id,
        rating_type=rating_type,
        allow_negative_discharge=(
            args.allow_negative_discharge
        ),
    )

    rating_checked = validate_rating_monotonicity(
        rating
    )

    inverse_rating = create_inverse_rating_table(
        rating
    )

    rating_output_path = (
        PROCESSED_OUTPUT_DIR
        / f"{site_id}_stage_to_discharge_rating.csv"
    )

    inverse_output_path = (
        PROCESSED_OUTPUT_DIR
        / f"{site_id}_discharge_to_stage_rating.csv"
    )

    diagnostic_output_path = (
        PROCESSED_OUTPUT_DIR
        / f"{site_id}_rating_monotonicity_check.csv"
    )

    metadata_output_path = (
        PROCESSED_OUTPUT_DIR
        / f"{site_id}_rating_metadata.json"
    )

    rating.to_csv(
        rating_output_path,
        index=False,
    )

    inverse_rating.to_csv(
        inverse_output_path,
        index=False,
    )

    rating_checked.to_csv(
        diagnostic_output_path,
        index=False,
    )

    summary = build_rating_summary(
        rating=rating,
        asset=selected_asset,
        site_id=site_id,
        nwm_reach_id=args.nwm_reach_id,
    )

    summary["rdb_comment_metadata"] = metadata

    metadata_output_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    plot_rating_curve(
        rating=rating,
        site_id=site_id,
    )

    plot_rating_curve_log_discharge(
        rating=rating,
        site_id=site_id,
    )

    print("\nRating-curve summary:")

    print(
        f"  USGS site:             {site_id}"
    )

    print(
        f"  NWM ReachID:           {args.nwm_reach_id}"
    )

    print(
        f"  Rating type:           {rating_type}"
    )

    print(
        f"  Number of points:      {len(rating):,}"
    )

    print(
        "  Gauge-height range:    "
        f"{rating['gauge_height_ft'].min():,.3f} to "
        f"{rating['gauge_height_ft'].max():,.3f} ft"
    )

    print(
        "  Discharge range:       "
        f"{rating['discharge_cfs'].min():,.3f} to "
        f"{rating['discharge_cfs'].max():,.3f} ft³/s"
    )

    print("\nOutputs created:")

    print(
        f"  Original RDB:          {raw_output_path}"
    )

    print(
        f"  Stage -> discharge:    {rating_output_path}"
    )

    print(
        f"  Discharge -> stage:    {inverse_output_path}"
    )

    print(
        f"  Monotonicity check:    {diagnostic_output_path}"
    )

    print(
        f"  Metadata:              {metadata_output_path}"
    )

    print(
        f"  Figures:               {FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()