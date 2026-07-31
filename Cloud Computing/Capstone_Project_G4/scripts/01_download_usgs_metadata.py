from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GAUGE_CONFIG_PATH = PROJECT_ROOT / "config" / "gauges.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "usgs_gauge_metadata.csv"


def load_site_ids(config_path: Path) -> list[str]:
    """Load included USGS gauge identifiers from the project configuration."""

    gauges = pd.read_csv(config_path, dtype={"site_id": str})

    if "include" in gauges.columns:
        include_mask = (
            gauges["include"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "y"})
        )
        gauges = gauges.loc[include_mask]

    site_ids = gauges["site_id"].dropna().astype(str).str.zfill(8).tolist()

    if not site_ids:
        raise ValueError(f"No included gauges were found in {config_path}")

    return site_ids


def get_usgs_site_metadata(site_ids: list[str]) -> pd.DataFrame:
    """Download monitoring-location metadata from the USGS OGC API."""

    identifiers = ",".join(f"USGS-{site_id}" for site_id in site_ids)

    url = (
        "https://api.waterdata.usgs.gov/ogcapi/v0/"
        "collections/monitoring-locations/items"
    )

    params = {
        "f": "json",
        "id": identifiers,
        "limit": len(site_ids),
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    geojson = response.json()

    rows = []

    for feature in geojson.get("features", []):
        properties = feature["properties"]
        longitude, latitude = feature["geometry"]["coordinates"]

        rows.append(
            {
                "site_id": properties.get("monitoring_location_number"),
                "site_name": properties.get("monitoring_location_name"),
                "latitude": latitude,
                "longitude": longitude,
                "site_type": properties.get("site_type"),
                "state": properties.get("state_name"),
                "county": properties.get("county_name"),
                "huc": properties.get("hydrologic_unit_code"),
                "basin_code": properties.get("basin_code"),
                "altitude_ft": properties.get("altitude"),
                "vertical_datum": properties.get("vertical_datum_name"),
                "drainage_area_sqmi": properties.get("drainage_area"),
                "contributing_area_sqmi": properties.get(
                    "contributing_drainage_area"
                ),
                "time_zone": properties.get("time_zone_abbreviation"),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    site_ids = load_site_ids(GAUGE_CONFIG_PATH)
    metadata = get_usgs_site_metadata(site_ids)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(OUTPUT_PATH, index=False)

    print(f"Downloaded metadata for {len(metadata)} gauges.")
    print(f"Saved metadata to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()