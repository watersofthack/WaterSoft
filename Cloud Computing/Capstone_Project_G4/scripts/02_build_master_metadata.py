from __future__ import annotations
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
USGS_METADATA_PATH = (
    PROJECT_ROOT / "data" / "interim" / "usgs_gauge_metadata.csv"
)

MASTER_METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "master_gauge_metadata.csv"
)
INPUT_FILE = USGS_METADATA_PATH
OUTPUT_FILE = MASTER_METADATA_PATH


# ---------------------------------------------------------------------
# Project-specific metadata collected from NOAA NWPS
#
# None means the value is unavailable or not applicable.
# Keep station and reach IDs as strings.
# ---------------------------------------------------------------------
NOAA_METADATA = [
    {
        "site_id": "08193000",
        "reach_id": "10893541",
        "action_stage_ft": 18.0,
        "minor_flood_stage_ft": 20.0,
        "moderate_flood_stage_ft": 24.0,
        "major_flood_stage_ft": 27.0,
        "gauge_status": "active",
        "notes": "",
    },
    {
        "site_id": "08194000",
        "reach_id": "10630401",
        "action_stage_ft": 9.0,
        "minor_flood_stage_ft": 15.0,
        "moderate_flood_stage_ft": 15.0,
        "major_flood_stage_ft": 17.0,
        "gauge_status": "active",
        "notes": (
            "NOAA reports the same threshold for minor and "
            "moderate flooding."
        ),
    },
    {
        "site_id": "08194500",
        "reach_id": "10631613",
        "action_stage_ft": 11.0,
        "minor_flood_stage_ft": 14.0,
        "moderate_flood_stage_ft": 16.0,
        "major_flood_stage_ft": 19.0,
        "gauge_status": "active",
        "notes": "",
    },
    {
        "site_id": "08205500",
        "reach_id": "10661028",
        "action_stage_ft": 6.0,
        "minor_flood_stage_ft": 6.0,
        "moderate_flood_stage_ft": 7.0,
        "major_flood_stage_ft": 17.0,
        "gauge_status": "active",
        "notes": (
            "NOAA reports the same threshold for action and "
            "minor flooding."
        ),
    },
    {
        "site_id": "08206600",
        "reach_id": "10664196",
        "action_stage_ft": 12.0,
        "minor_flood_stage_ft": 22.0,
        "moderate_flood_stage_ft": 26.0,
        "major_flood_stage_ft": 27.0,
        "gauge_status": "active",
        "notes": "",
    },
    {
        "site_id": "08206700",
        "reach_id": "10671061",
        "action_stage_ft": 12.0,
        "minor_flood_stage_ft": 21.0,
        "moderate_flood_stage_ft": 23.0,
        "major_flood_stage_ft": 26.0,
        "gauge_status": "active",
        "notes": "",
    },
    {
        "site_id": "08206900",
        "reach_id": None,
        "action_stage_ft": None,
        "minor_flood_stage_ft": None,
        "moderate_flood_stage_ft": None,
        "major_flood_stage_ft": None,
        "gauge_status": "active",
        "notes": "No flood-stage category data shown on NOAA NWPS.",
    },
    {
        "site_id": "08208000",
        "reach_id": "10828654",
        "action_stage_ft": 20.0,
        "minor_flood_stage_ft": 20.0,
        "moderate_flood_stage_ft": 24.0,
        "major_flood_stage_ft": 26.0,
        "gauge_status": "active",
        "notes": (
            "NOAA reports the same threshold for action and "
            "minor flooding."
        ),
    },
    {
        "site_id": "08210000",
        "reach_id": "3168766",
        "action_stage_ft": 20.0,
        "minor_flood_stage_ft": 25.0,
        "moderate_flood_stage_ft": 27.0,
        "major_flood_stage_ft": 35.0,
        "gauge_status": "active",
        "notes": "",
    },
    {
        "site_id": "08210100",
        "reach_id": None,
        "action_stage_ft": None,
        "minor_flood_stage_ft": None,
        "moderate_flood_stage_ft": None,
        "major_flood_stage_ft": None,
        "gauge_status": "discontinued",
        "notes": "Gauge discontinued after 2010.",
    },
]

def infer_river_name(site_name: str | None) -> str | None:
    """Infer a standardized river or waterbody name."""
    if not isinstance(site_name, str):
        return None

    name = site_name.lower()

    if "nueces" in name:
        return "Nueces River"
    if "frio" in name:
        return "Frio River"
    if "atascosa" in name:
        return "Atascosa River"
    if "san miguel" in name:
        return "San Miguel Creek"
    if "choke canyon" in name:
        return "Choke Canyon Reservoir"

    return None


def infer_site_category(site_name: str | None) -> str | None:
    """Classify each location as a stream, reservoir, or other site."""
    if not isinstance(site_name, str):
        return None

    name = site_name.lower()

    if "res " in name or "reservoir" in name:
        return "Reservoir"

    if (
        " rv " in name
        or "river" in name
        or " ck " in name
        or "creek" in name
    ):
        return "Stream"

    return "Other"


def load_usgs_metadata(path: Path) -> pd.DataFrame:
    """Load downloaded USGS site metadata."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(
        path,
        dtype={
            "site_id": str,
            "huc": str,
        },
    )

    if "site_id" not in df.columns:
        raise ValueError("Input CSV must contain a 'site_id' column.")

    # Preserve the leading zero in USGS station IDs.
    df["site_id"] = (
        df["site_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )

    # Preserve HUC leading zeros, if applicable.
    if "huc" in df.columns:
        df["huc"] = (
            df["huc"]
            .astype("string")
            .str.replace(r"\.0$", "", regex=True)
        )

    return df


def build_master_table(
    usgs_metadata: pd.DataFrame,
    noaa_records: list[dict],
) -> pd.DataFrame:
    """Merge USGS metadata with NOAA project metadata."""

    noaa_df = pd.DataFrame(noaa_records)

    noaa_df["site_id"] = (
        noaa_df["site_id"]
        .astype(str)
        .str.strip()
        .str.zfill(8)
    )

    noaa_df["reach_id"] = noaa_df["reach_id"].astype("string")

    # Ensure IDs are unique before merging.
    if usgs_metadata["site_id"].duplicated().any():
        duplicates = usgs_metadata.loc[
            usgs_metadata["site_id"].duplicated(keep=False),
            "site_id",
        ].tolist()

        raise ValueError(
            "Duplicate USGS IDs in downloaded metadata: "
            + ", ".join(sorted(set(duplicates)))
        )

    if noaa_df["site_id"].duplicated().any():
        raise ValueError("Duplicate site IDs found in NOAA_METADATA.")

    master = usgs_metadata.merge(
        noaa_df,
        on="site_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    # Show any station that was found in only one input source.
    unmatched = master.loc[
        master["_merge"] != "both",
        ["site_id", "_merge"],
    ]

    if not unmatched.empty:
        print("\nWarning: unmatched station records:")
        print(unmatched.to_string(index=False))

    master = master.drop(columns="_merge")

    # Infer river names from the official USGS site name.
    master["river"] = master["site_name"].apply(infer_river_name)

    master["site_category"] = master["site_name"].apply(

        infer_site_category

    )

    # Useful project flags.
    master["has_reach_id"] = master["reach_id"].notna()

    threshold_columns = [
        "action_stage_ft",
        "minor_flood_stage_ft",
        "moderate_flood_stage_ft",
        "major_flood_stage_ft",
    ]

    master["has_flood_categories"] = master[
        threshold_columns
    ].notna().any(axis=1)

    # Check whether thresholds are nondecreasing.
    def thresholds_are_valid(row: pd.Series) -> bool | None:
        values = [
            row[column]
            for column in threshold_columns
            if pd.notna(row[column])
        ]

        if len(values) < 2:
            return None

        return all(
            current <= following
            for current, following in zip(values, values[1:])
        )

    master["flood_thresholds_valid"] = master.apply(
        thresholds_are_valid,
        axis=1,
    )

    preferred_columns = [
        "site_id",
        "reach_id",
        "site_name",
        "river",
        "site_category",
        "latitude",
        "longitude",
        "site_type",
        "state",
        "county",
        "huc",
        "drainage_area_sqmi",
        "contributing_area_sqmi",
        "altitude_ft",
        "vertical_datum",
        "time_zone",
        "action_stage_ft",
        "minor_flood_stage_ft",
        "moderate_flood_stage_ft",
        "major_flood_stage_ft",
        "gauge_status",
        "has_reach_id",
        "has_flood_categories",
        "flood_thresholds_valid",
        "notes",
    ]

    existing = [
        column
        for column in preferred_columns
        if column in master.columns
    ]

    extra = [
        column
        for column in master.columns
        if column not in existing
    ]

    master = master[existing + extra]

    return master.sort_values(
        by=["river", "site_id"],
        na_position="last",
    ).reset_index(drop=True)

def main() -> None:
    usgs_metadata = load_usgs_metadata(USGS_METADATA_PATH)

    master = build_master_table(
        usgs_metadata=usgs_metadata,
        noaa_records=NOAA_METADATA,
    )

    MASTER_METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    master.to_csv(
        MASTER_METADATA_PATH,
        index=False,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    display_columns = [
        "site_id",
        "reach_id",
        "site_name",
        "river",
        "site_category",
        "drainage_area_sqmi",
        "action_stage_ft",
        "minor_flood_stage_ft",
        "moderate_flood_stage_ft",
        "major_flood_stage_ft",
        "gauge_status",
    ]

    print("\nMaster gauge metadata:")
    print(master[display_columns].to_string(index=False))

    print(
        "\nSaved master table to:"
        f"\n{MASTER_METADATA_PATH}"
    )


if __name__ == "__main__":
    main()