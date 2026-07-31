from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "master_gauge_metadata.csv"
)

TIMESERIES_DIR = PROJECT_ROOT / "data" / "raw" / "usgs" / "timeseries"

LONG_OUTPUT_PATH = (
    TIMESERIES_DIR / "usgs_gauge_height_streamflow_long.csv"
)

WIDE_OUTPUT_PATH = (
    TIMESERIES_DIR / "usgs_gauge_height_streamflow_wide.csv"
)

SUMMARY_OUTPUT_PATH = (
    TIMESERIES_DIR / "usgs_download_summary.csv"
)


# ---------------------------------------------------------------------
# Download configuration
# ---------------------------------------------------------------------
START_DATE = "2010-01-01"
# END_DATE = "2024-12-31"
END_DATE = date.today().isoformat()

# USGS parameter codes:
# 00060 = discharge/streamflow, cubic feet per second
# 00065 = gage height, feet
PARAMETER_CODES = ["00060", "00065"]

USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"

# Request one calendar year at a time.
# Chunking reduces the chance of oversized requests and makes retries easier.
CHUNK_FREQUENCY = "YS"

REQUEST_TIMEOUT_SECONDS = 120
MAX_RETRIES = 4
RETRY_WAIT_SECONDS = 5

# These downloads are network-bound, so threads are more useful than
# multiprocessing. Keep this modest to avoid overwhelming the USGS service.
MAX_DOWNLOAD_WORKERS = 4

# Select which stations to download.
ACTIVE_ONLY = True
STREAM_SITES_ONLY = True

# A discontinued station can still have historical data.
# Set this to True to include it despite ACTIVE_ONLY.
INCLUDE_DISCONTINUED_HISTORICAL = False

# Reservoir data can be useful later, but it is excluded from the first
# upstream/downstream stream-gauge experiment by default.
INCLUDE_RESERVOIRS = False


PARAMETER_NAMES = {
    "00060": "streamflow_cfs",
    "00065": "gage_height_ft",
}


def normalize_site_id(series: pd.Series) -> pd.Series:
    """Preserve leading zeros in eight-digit USGS station IDs."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )


def load_selected_gauges(path: Path) -> pd.DataFrame:
    """Load the master metadata table and select gauges for downloading."""
    if not path.exists():
        raise FileNotFoundError(
            f"Master metadata file was not found:\n{path}"
        )

    metadata = pd.read_csv(
        path,
        dtype={
            "site_id": str,
            "reach_id": str,
        },
    )

    if "site_id" not in metadata.columns:
        raise ValueError(
            "The metadata CSV must contain a 'site_id' column."
        )

    metadata["site_id"] = normalize_site_id(metadata["site_id"])

    selected = metadata.copy()

    if ACTIVE_ONLY and "gauge_status" in selected.columns:
        if INCLUDE_DISCONTINUED_HISTORICAL:
            pass
        else:
            selected = selected.loc[
                selected["gauge_status"].str.lower().eq("active")
            ]

    if STREAM_SITES_ONLY and "site_category" in selected.columns:
        if INCLUDE_RESERVOIRS:
            selected = selected.loc[
                selected["site_category"].isin(
                    ["Stream", "Reservoir"]
                )
            ]
        else:
            selected = selected.loc[
                selected["site_category"].eq("Stream")
            ]

    selected = selected.drop_duplicates(
        subset="site_id"
    ).reset_index(drop=True)

    if selected.empty:
        raise ValueError(
            "No gauges remain after applying the selection filters."
        )

    return selected


def build_date_chunks(
    start_date: str,
    end_date: str,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Divide a date range into calendar-year request chunks."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError("START_DATE must be before END_DATE.")

    boundaries = pd.date_range(
        start=start.normalize(),
        end=end.normalize(),
        freq=CHUNK_FREQUENCY,
    )

    chunk_starts = [start]

    for boundary in boundaries:
        if start < boundary <= end:
            chunk_starts.append(boundary)

    chunk_starts = sorted(set(chunk_starts))

    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for index, chunk_start in enumerate(chunk_starts):
        if index + 1 < len(chunk_starts):
            chunk_end = chunk_starts[index + 1] - pd.Timedelta(days=1)
        else:
            chunk_end = end

        chunks.append((chunk_start, min(chunk_end, end)))

    return chunks


def request_usgs_json(
    site_ids: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, Any]:
    """Request USGS instantaneous values with retry handling."""
    params = {
        "format": "json",
        "sites": ",".join(site_ids),
        "parameterCd": ",".join(PARAMETER_CODES),
        "startDT": start_date.strftime("%Y-%m-%d"),
        "endDT": end_date.strftime("%Y-%m-%d"),
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

        except (
            requests.RequestException,
            ValueError,
        ) as exc:
            last_error = exc

            print(
                f"  Request attempt {attempt}/{MAX_RETRIES} failed: "
                f"{exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS * attempt)

    raise RuntimeError(
        "USGS request failed after all retry attempts."
    ) from last_error


def extract_site_id(source_info: dict[str, Any]) -> str | None:
    """Extract and normalize a USGS site ID from WaterML JSON."""
    site_codes = (
        source_info
        .get("siteCode", [])
    )

    if not site_codes:
        return None

    site_id = site_codes[0].get("value")

    if site_id is None:
        return None

    return str(site_id).strip().zfill(8)


def extract_parameter_code(
    variable: dict[str, Any],
) -> str | None:
    """Extract the five-digit parameter code."""
    codes = variable.get("variableCode", [])

    if not codes:
        return None

    code = codes[0].get("value")

    if code is None:
        return None

    return str(code).strip().zfill(5)


def parse_usgs_response(
    payload: dict[str, Any],
) -> pd.DataFrame:
    """Convert USGS WaterML JSON into a tidy dataframe."""
    time_series = (
        payload
        .get("value", {})
        .get("timeSeries", [])
    )

    records: list[dict[str, Any]] = []

    for series in time_series:
        source_info = series.get("sourceInfo", {})
        variable = series.get("variable", {})

        site_id = extract_site_id(source_info)
        parameter_code = extract_parameter_code(variable)

        if site_id is None or parameter_code is None:
            continue

        parameter_name = PARAMETER_NAMES.get(
            parameter_code,
            variable.get("variableDescription", parameter_code),
        )

        unit_code = (
            variable
            .get("unit", {})
            .get("unitCode")
        )

        site_name = source_info.get("siteName")

        value_blocks = series.get("values", [])

        for value_block in value_blocks:
            for observation in value_block.get("value", []):
                raw_value = observation.get("value")
                timestamp = observation.get("dateTime")
                qualifiers = observation.get("qualifiers", [])

                records.append(
                    {
                        "site_id": site_id,
                        "site_name_api": site_name,
                        "datetime": timestamp,
                        "parameter_code": parameter_code,
                        "parameter_name": parameter_name,
                        "value": raw_value,
                        "unit": unit_code,
                        "qualifiers": ",".join(qualifiers),
                    }
                )

    if not records:
        return pd.DataFrame(
            columns=[
                "site_id",
                "site_name_api",
                "datetime",
                "parameter_code",
                "parameter_name",
                "value",
                "unit",
                "qualifiers",
            ]
        )

    data = pd.DataFrame(records)

    data["site_id"] = normalize_site_id(data["site_id"])

    data["datetime"] = pd.to_datetime(
        data["datetime"],
        utc=True,
        errors="coerce",
    )

    data["value"] = pd.to_numeric(
        data["value"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["site_id", "datetime", "value"]
    )

    return data


def download_date_chunk(
    site_ids: list[str],
    chunk_start: pd.Timestamp,
    chunk_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download and parse one date chunk."""
    try:
        payload = request_usgs_json(
            site_ids=site_ids,
            start_date=chunk_start,
            end_date=chunk_end,
        )

        chunk_data = parse_usgs_response(payload)
        sites_returned = (
            chunk_data["site_id"].nunique()
            if not chunk_data.empty
            else 0
        )

        summary = {
            "start_date": chunk_start.date().isoformat(),
            "end_date": chunk_end.date().isoformat(),
            "status": "success",
            "rows_downloaded": len(chunk_data),
            "sites_returned": sites_returned,
            "error": "",
        }
    except Exception as exc:
        chunk_data = pd.DataFrame()
        summary = {
            "start_date": chunk_start.date().isoformat(),
            "end_date": chunk_end.date().isoformat(),
            "status": "failed",
            "rows_downloaded": 0,
            "sites_returned": 0,
            "error": str(exc),
        }

    return chunk_data, summary


def download_all_data(
    gauges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download all sites and return data plus a request summary."""
    site_ids = gauges["site_id"].tolist()

    chunks = build_date_chunks(
        start_date=START_DATE,
        end_date=END_DATE,
    )

    downloaded_frames: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []

    print(f"Selected {len(site_ids)} gauges:")
    print(", ".join(site_ids))

    print(
        f"\nDownloading {START_DATE} through {END_DATE} "
        f"in {len(chunks)} date chunks with up to "
        f"{MAX_DOWNLOAD_WORKERS} parallel workers."
    )

    workers = min(MAX_DOWNLOAD_WORKERS, len(chunks))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_chunk = {
            executor.submit(
                download_date_chunk,
                site_ids,
                chunk_start,
                chunk_end,
            ): (chunk_number, chunk_start, chunk_end)
            for chunk_number, (chunk_start, chunk_end) in enumerate(
                chunks,
                start=1,
            )
        }

        for completed_number, future in enumerate(
            as_completed(future_to_chunk),
            start=1,
        ):
            chunk_number, chunk_start, chunk_end = future_to_chunk[future]
            chunk_data, summary = future.result()

            if not chunk_data.empty:
                downloaded_frames.append(chunk_data)

            summary_records.append(summary)

            progress = (
                f"[{completed_number}/{len(chunks)} completed; "
                f"chunk {chunk_number}] "
                f"{chunk_start.date()} to {chunk_end.date()}"
            )

            if summary["status"] == "success":
                print(
                    f"{progress}: "
                    f"{summary['rows_downloaded']:,} observations "
                    f"for {summary['sites_returned']} sites."
                )
            else:
                print(f"{progress}: failed: {summary['error']}")

    if downloaded_frames:
        data = pd.concat(
            downloaded_frames,
            ignore_index=True,
        )
    else:
        data = pd.DataFrame()

    summary = (
        pd.DataFrame(summary_records)
        .sort_values("start_date")
        .reset_index(drop=True)
    )

    return data, summary


def attach_master_metadata(
    data: pd.DataFrame,
    gauges: pd.DataFrame,
) -> pd.DataFrame:
    """Attach selected metadata fields to downloaded observations."""
    if data.empty:
        return data

    metadata_columns = [
        column
        for column in [
            "site_id",
            "site_name",
            "river",
            "site_category",
            "reach_id",
            "latitude",
            "longitude",
            "drainage_area_sqmi",
            "action_stage_ft",
            "minor_flood_stage_ft",
            "moderate_flood_stage_ft",
            "major_flood_stage_ft",
            "gauge_status",
        ]
        if column in gauges.columns
    ]

    return data.merge(
        gauges[metadata_columns],
        on="site_id",
        how="left",
        validate="many_to_one",
    )


def create_wide_table(data: pd.DataFrame) -> pd.DataFrame:
    """Create one row per site and timestamp with separate variables."""
    if data.empty:
        return pd.DataFrame()

    duplicate_mask = data.duplicated(
        subset=[
            "site_id",
            "datetime",
            "parameter_name",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())

        print(
            f"\nWarning: {duplicate_count:,} duplicate "
            "site-time-parameter records were found."
        )
        print(
            "The wide table will retain the first record for each "
            "site-time-parameter combination."
        )

    unique_data = data.drop_duplicates(
        subset=[
            "site_id",
            "datetime",
            "parameter_name",
        ],
        keep="first",
    )

    wide = (
        unique_data.pivot(
            index=["site_id", "datetime"],
            columns="parameter_name",
            values="value",
        )
        .reset_index()
    )

    wide.columns.name = None

    metadata_columns = [
        column
        for column in [
            "site_name",
            "river",
            "site_category",
            "reach_id",
            "latitude",
            "longitude",
            "drainage_area_sqmi",
            "action_stage_ft",
            "minor_flood_stage_ft",
            "moderate_flood_stage_ft",
            "major_flood_stage_ft",
            "gauge_status",
        ]
        if column in data.columns
    ]

    site_metadata = (
        data[["site_id"] + metadata_columns]
        .drop_duplicates(subset="site_id")
    )

    wide = wide.merge(
        site_metadata,
        on="site_id",
        how="left",
        validate="many_to_one",
    )

    preferred = [
        "site_id",
        "site_name",
        "river",
        "site_category",
        "reach_id",
        "datetime",
        "streamflow_cfs",
        "gage_height_ft",
        "latitude",
        "longitude",
        "drainage_area_sqmi",
        "action_stage_ft",
        "minor_flood_stage_ft",
        "moderate_flood_stage_ft",
        "major_flood_stage_ft",
        "gauge_status",
    ]

    existing = [
        column for column in preferred
        if column in wide.columns
    ]

    remaining = [
        column for column in wide.columns
        if column not in existing
    ]

    return (
        wide[existing + remaining]
        .sort_values(["site_id", "datetime"])
        .reset_index(drop=True)
    )


def print_data_summary(data: pd.DataFrame) -> None:
    """Print basic availability statistics."""
    if data.empty:
        print("\nNo observations were downloaded.")
        return

    summary = (
        data.groupby(
            ["site_id", "parameter_name"],
            dropna=False,
        )
        .agg(
            first_datetime=("datetime", "min"),
            last_datetime=("datetime", "max"),
            observation_count=("value", "count"),
            missing_count=("value", lambda x: x.isna().sum()),
        )
        .reset_index()
    )

    print("\nDownloaded-data availability:")
    print(summary.to_string(index=False))

def main() -> None:
    """Download USGS gauge observations and save project datasets."""

    # Ensure the output directory exists before writing files.
    TIMESERIES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    gauges = load_selected_gauges(
        MASTER_METADATA_PATH
    )

    data, request_summary = download_all_data(gauges)

    request_summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    if data.empty:
        print(
            "\nNo observations were returned. "
            "Check the dates, site selection, and parameter availability."
        )
        print(
            f"\nRequest summary saved to:\n{SUMMARY_OUTPUT_PATH}"
        )
        return

    data = attach_master_metadata(
        data=data,
        gauges=gauges,
    )

    data = (
        data.sort_values(
            [
                "site_id",
                "datetime",
                "parameter_code",
            ]
        )
        .drop_duplicates(
            subset=[
                "site_id",
                "datetime",
                "parameter_code",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    data.to_csv(
        LONG_OUTPUT_PATH,
        index=False,
    )

    wide = create_wide_table(data)

    wide.to_csv(
        WIDE_OUTPUT_PATH,
        index=False,
    )

    print_data_summary(data)

    print("\nSaved files:")
    print(
        f"\nLong-format observations:\n{LONG_OUTPUT_PATH}"
    )
    print(
        f"\nWide-format observations:\n{WIDE_OUTPUT_PATH}"
    )
    print(
        f"\nRequest summary:\n{SUMMARY_OUTPUT_PATH}"
    )

if __name__ == "__main__":
    main()
