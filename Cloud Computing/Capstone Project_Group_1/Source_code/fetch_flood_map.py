#!/usr/bin/env python3
"""
Fetch a REAL-TIME flood map for the eastern United States.

Two authoritative, key-free NOAA/NWS sources are combined:

1. NWS active flood alerts  (api.weather.gov/alerts/active)
   -> flood-warning / flash-flood / coastal-flood polygons, with CAP severity.
2. NWPS river gauges        (api.water.noaa.gov/nwps/v1/gauges)
   -> gauge points whose observed `floodCategory` is action/minor/moderate/major.

Both are distilled into a common list of "flood anchors" (a point, a severity
0-4, and a radius). `generate_reports.py --anchors flood_anchors.csv` then places
synthetic iOS reports at these real flooded locations, so the dataset matches the
live NWS-FIM / NWPS picture instead of arbitrary coordinates.

Outputs
-------
  flood_map.geojson    Alert polygons + gauge points (open in any GIS / geojson.io).
  flood_anchors.csv    Distilled anchors the generator consumes.

Fallback
--------
On a dry day with no active flooding, falls back to the most elevated gauges, and
finally to a small built-in list of chronic eastern-US flood locations, so the
downstream generator always has something to anchor to. The chosen mode is printed.

Stdlib only. NWS requires a descriptive User-Agent (set below).
"""

import argparse
import csv
import json
import math
import time
import urllib.parse
import urllib.request

UA = "FloodOps-WaterSoftHack/1.0 (research prototype; contact kahriziehsan490@gmail.com)"

# Whole-US coverage. Gauge queries are TILED because a single US-wide bbox
# 504s on the NWPS server. Tiles: (xmin, ymin, xmax, ymax) in lon/lat.
US_TILES = [
    (-125, 40, -105, 50), (-105, 40, -85, 50), (-85, 40, -66, 50),  # north band
    (-125, 24, -105, 40), (-105, 24, -85, 40), (-85, 24, -66, 40),  # south band
]

# States included for NWS alerts: contiguous 48 + DC (Alaska, Hawaii, PR excluded).
US_ALERT_AREAS = [
    "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "IA", "ID", "IL",
    "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC",
    "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
]

FLOOD_EVENTS = ["Flood Warning", "Flash Flood Warning", "Coastal Flood Warning",
                "Flood Advisory", "Flash Flood Watch", "Flood Watch",
                "Coastal Flood Advisory"]

# event type -> base severity class 0..4
EVENT_SEV = {
    "Flash Flood Warning": 4, "Flood Warning": 3, "Coastal Flood Warning": 3,
    "Flood Advisory": 2, "Coastal Flood Advisory": 2,
    "Flash Flood Watch": 2, "Flood Watch": 1, "Coastal Flood Watch": 1,
}
CAP_BUMP = {"Extreme": 1, "Severe": 0, "Moderate": -1, "Minor": -1, "Unknown": 0}
GAUGE_SEV = {"major": 4, "moderate": 3, "minor": 2, "action": 1, "no_flooding": 0}

# Chronic eastern-US flood locations used only as a last-resort fallback.
FALLBACK = [
    ("Charleston", "SC", 32.7765, -79.9311, 2),
    ("Norfolk", "VA", 36.8508, -76.2859, 2),
    ("Ellicott City", "MD", 39.2673, -76.7983, 3),
    ("Asheville", "NC", 35.5951, -82.5515, 3),
    ("Miami", "FL", 25.7617, -80.1918, 2),
    ("Philadelphia", "PA", 39.9526, -75.1652, 2),
]

# FEMA National Flood Hazard Layer (NFHL) -- static Special Flood Hazard Areas.
FEMA_NFHL_QUERY = ("https://hazards.fema.gov/arcgis/rest/services/public/"
                   "NFHL/MapServer/28/query")
FEMA_HIGH_RISK = ("A", "AE", "AO", "AH", "VE", "V", "AR", "A99")  # SFHA zones

# NWS-FIM: Stage-Based CatFIM library -- REAL inundation-extent polygons per gauge
# per flood category. floodCategory -> the matching threshold layer id.
CATFIM_BASE = ("https://maps.water.noaa.gov/server/rest/services/"
               "fim_libs/static_stage_based_catfim/FeatureServer")
CATFIM_LAYER = {"action": 3, "minor": 6, "moderate": 9, "major": 12}

# Seed cities to sample FEMA flood-prone zones around (nationwide metros).
SEED_CITIES = [
    # East
    ("Charleston", "SC", 32.7765, -79.9311),
    ("Norfolk", "VA", 36.8508, -76.2859),
    ("Miami", "FL", 25.7617, -80.1918),
    ("Tampa", "FL", 27.9506, -82.4572),
    ("Ellicott City", "MD", 39.2673, -76.7983),
    ("Philadelphia", "PA", 39.9526, -75.1652),
    ("New York", "NY", 40.7128, -74.0060),
    ("Boston", "MA", 42.3601, -71.0589),
    # Central / Gulf
    ("Houston", "TX", 29.7604, -95.3698),
    ("New Orleans", "LA", 29.9511, -90.0715),
    ("Dallas", "TX", 32.7767, -96.7970),
    ("Nashville", "TN", 36.1627, -86.7816),
    ("St. Louis", "MO", 38.6270, -90.1994),
    ("Chicago", "IL", 41.8781, -87.6298),
    ("Minneapolis", "MN", 44.9778, -93.2650),
    ("Kansas City", "MO", 39.0997, -94.5786),
    # West
    ("Denver", "CO", 39.7392, -104.9903),
    ("Phoenix", "AZ", 33.4484, -112.0740),
    ("Los Angeles", "CA", 34.0522, -118.2437),
    ("Sacramento", "CA", 38.5816, -121.4944),
    ("Seattle", "WA", 47.6062, -122.3321),
    ("Portland", "OR", 45.5152, -122.6784),
    ("Salt Lake City", "UT", 40.7608, -111.8910),
]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def http_json(url, timeout=40, retries=3):
    """GET JSON with retry+backoff on rate-limit (429) and 5xx errors."""
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept": "application/geo+json, application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = 3 * (attempt + 1)
                print(f"  ! {e.code}; retrying in {wait}s ...")
                time.sleep(wait)
                continue
            print(f"  ! request failed: {e}")
            return None
        except Exception as e:  # noqa: BLE001 -- network is best-effort
            print(f"  ! request failed: {e}")
            return None
    return None


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Source 1: NWS active flood alerts
# --------------------------------------------------------------------------- #

def poly_rings(geom):
    """Yield coordinate rings (list of [lon,lat]) for Polygon/MultiPolygon."""
    if not geom:
        return
    if geom["type"] == "Polygon":
        yield geom["coordinates"][0]
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            yield poly[0]


def centroid_radius(ring):
    lon = sum(p[0] for p in ring) / len(ring)
    lat = sum(p[1] for p in ring) / len(ring)
    rad = max(haversine_km(lat, lon, p[1], p[0]) for p in ring)
    return lat, lon, rad


def fetch_alerts(states=None):
    params = {"status": "actual", "message_type": "alert",
              "event": ",".join(FLOOD_EVENTS)}
    if states:                       # omit -> nationwide
        params["area"] = ",".join(states)
    url = "https://api.weather.gov/alerts/active?" + urllib.parse.urlencode(params)
    scope = f"{len(states)} states" if states else "nationwide"
    print(f"Fetching NWS active flood alerts ({scope}) ...")
    data = http_json(url)
    anchors, features, skipped = [], [], 0
    if not data:
        return anchors, features
    for f in data.get("features", []):
        p = f.get("properties", {})
        geom = f.get("geometry")
        if not geom:
            skipped += 1
            continue
        event = p.get("event", "")
        sev = clamp(EVENT_SEV.get(event, 2) + CAP_BUMP.get(p.get("severity", "Unknown"), 0), 1, 4)
        for ring in poly_rings(geom):
            lat, lon, rad = centroid_radius(ring)
            anchors.append({
                "anchor_id": p.get("id", "")[-16:] or f"nws_{len(anchors)}",
                "source": "nws_alert", "name": (p.get("areaDesc") or "")[:60],
                "state": (p.get("areaDesc") or "").split(",")[-1].strip()[:2],
                "lat": round(lat, 5), "lon": round(lon, 5),
                "severity_class": sev, "radius_km": round(clamp(rad, 2, 30), 1),
                "event_type": event, "valid_time": p.get("onset") or p.get("sent", ""),
                "gauge_id": "", "gauge_stage_ft": "",
            })
        p["source"] = "nws_alert"          # tag for the visualizer styling
        p["severity_class"] = sev
        features.append(f)  # keep original polygon for the geojson
    print(f"  alerts: {len(features)} polygons, {len(anchors)} anchors "
          f"({skipped} zone-only alerts skipped)")
    return anchors, features


# --------------------------------------------------------------------------- #
# Source 2: NWPS river gauges in flood
# --------------------------------------------------------------------------- #

def _gauge_to_anchor(g, min_class):
    """Return (anchor, feature) for an in-flood gauge, or (None, None)."""
    obs = (g.get("status") or {}).get("observed") or {}
    cat = obs.get("floodCategory", "no_flooding")
    sev = GAUGE_SEV.get(cat, 0)
    if sev < min_class:
        return None, None
    lat, lon = g.get("latitude"), g.get("longitude")
    if lat is None or lon is None:
        return None, None
    stage = obs.get("primary")
    anchor = {
        "anchor_id": g.get("lid", ""), "source": "nwps_gauge",
        "name": (g.get("name") or "")[:60],
        "state": ((g.get("state") or {}).get("abbreviation") or "")[:2],
        "lat": round(lat, 5), "lon": round(lon, 5),
        "severity_class": sev, "radius_km": 5.0,
        "event_type": f"gauge_{cat}", "valid_time": obs.get("validTime", ""),
        "gauge_id": g.get("lid", ""),
        "gauge_stage_ft": stage if stage not in (None, -999) else "",
    }
    feature = {"type": "Feature",
               "geometry": {"type": "Point", "coordinates": [lon, lat]},
               "properties": {"source": "nwps_gauge", "lid": g.get("lid"),
                              "name": g.get("name"), "floodCategory": cat,
                              "severity_class": sev}}
    return anchor, feature


def fetch_gauges(tiles, min_class=1):
    """Fetch NWPS gauges over one or more bbox TILES and keep those in flood.

    The US-wide bbox 504s on the server, so coverage is split into tiles and the
    results merged (deduped by gauge id).
    """
    print(f"Fetching NWPS river gauges in flood ({len(tiles)} tiles) ...")
    anchors, features, seen = [], [], set()
    for i, (x0, y0, x1, y1) in enumerate(tiles):
        if i:
            time.sleep(1.0)          # be polite between tiles to avoid HTTP 429
        q = urllib.parse.urlencode({
            "bbox.xmin": x0, "bbox.ymin": y0, "bbox.xmax": x1, "bbox.ymax": y1,
            "srid": "EPSG_4326"})
        data = http_json("https://api.water.noaa.gov/nwps/v1/gauges?" + q, timeout=90)
        if not data:
            print(f"    tile {x0},{y0},{x1},{y1}: (skipped)")
            continue
        n0 = len(anchors)
        for g in data.get("gauges", []):
            lid = g.get("lid", "")
            if lid in seen:
                continue
            a, ft = _gauge_to_anchor(g, min_class)
            if a is None:
                continue
            seen.add(lid)
            anchors.append(a)
            features.append(ft)
        print(f"    tile {x0},{y0},{x1},{y1}: +{len(anchors) - n0} in flood")
    print(f"  gauges in flood (>= class {min_class}): {len(anchors)}")
    return anchors, features


# --------------------------------------------------------------------------- #
# Source 3: FEMA NFHL flood-prone zones (static Special Flood Hazard Areas)
# --------------------------------------------------------------------------- #

def fetch_fema_nfhl(seeds, per_seed=3, half_deg=0.06):
    """Sample FEMA NFHL Special Flood Hazard Area polygons around seed cities.

    Unlike the live sources, these are STATIC 100-yr floodplains -- flood-PRONE,
    not currently flooding. Useful as extra placement anchors (and on dry days).
    """
    print("Fetching FEMA NFHL flood-prone zones ...")
    where = "FLD_ZONE IN (" + ",".join(f"'{z}'" for z in FEMA_HIGH_RISK) + ")"
    anchors, features = [], []
    for name, state, lat, lon in seeds:
        env = f"{lon-half_deg},{lat-half_deg},{lon+half_deg},{lat+half_deg}"
        q = urllib.parse.urlencode({
            "geometry": env, "geometryType": "esriGeometryEnvelope",
            "inSR": "4326", "outSR": "4326",
            "spatialRel": "esriSpatialRelIntersects", "where": where,
            "outFields": "FLD_ZONE,ZONE_SUBTY", "returnGeometry": "true",
            "resultRecordCount": per_seed, "f": "geojson"})
        data = http_json(FEMA_NFHL_QUERY + "?" + q, timeout=45)
        if not data:
            continue
        for ft in data.get("features", [])[:per_seed]:
            geom = ft.get("geometry")
            zone = (ft.get("properties") or {}).get("FLD_ZONE", "")
            rings = list(poly_rings(geom))
            if not rings:
                continue
            la, lo, rad = centroid_radius(rings[0])
            sev = 2 if zone in ("VE", "V", "AE", "A") else 1  # prone, not live
            anchors.append({
                "anchor_id": f"fema_{name}_{zone}_{len(anchors)}",
                "source": "fema_nfhl", "name": f"{name} ({zone} zone)",
                "state": state, "lat": round(la, 5), "lon": round(lo, 5),
                "severity_class": sev, "radius_km": round(clamp(rad, 1, 10), 1),
                "event_type": f"fema_zone_{zone}", "valid_time": "static",
                "gauge_id": "", "gauge_stage_ft": "",
            })
            ft["properties"] = {"source": "fema_nfhl", "FLD_ZONE": zone,
                                "severity_class": sev, "city": name}
            features.append(ft)
    print(f"  FEMA NFHL flood-prone polygons: {len(anchors)}")
    return anchors, features


# --------------------------------------------------------------------------- #
# Source 4: NWS-FIM (CatFIM) real inundation-extent polygons
# --------------------------------------------------------------------------- #

def fetch_nws_fim(gauge_anchors, simplify_deg=0.0004):
    """For each gauge currently in flood, fetch the REAL inundation polygon at
    its current flood category from the NWS Stage-Based CatFIM library.

    This is the true 'flood = area' answer: the mapped water extent, not a point.
    Only gauges that have CatFIM coverage return a polygon; others are skipped.
    """
    print("Fetching NWS-FIM (CatFIM) inundation polygons ...")
    anchors, features = [], []
    for g in gauge_anchors:
        lid = (g.get("gauge_id") or "").lower()
        cat = g.get("event_type", "").replace("gauge_", "")
        layer = CATFIM_LAYER.get(cat)
        if not lid or layer is None:
            continue
        q = urllib.parse.urlencode({
            "where": f"ahps_lid='{lid}'",
            "outFields": "ahps_lid,name,magnitude,stage,stage_uni",
            "returnGeometry": "true", "outSR": "4326",
            "maxAllowableOffset": simplify_deg, "f": "geojson"})
        data = http_json(f"{CATFIM_BASE}/{layer}/query?" + q, timeout=45)
        if not data:
            continue
        for ft in data.get("features", []):
            geom = ft.get("geometry")
            rings = list(poly_rings(geom))
            if not rings:
                continue
            la, lo, rad = centroid_radius(rings[0])
            sev = GAUGE_SEV.get(cat, 2)
            props = ft.get("properties") or {}
            ft["properties"] = {"source": "nws_fim", "ahps_lid": lid,
                                "magnitude": cat, "severity_class": sev,
                                "name": props.get("name", g.get("name", "")),
                                "stage": props.get("stage", "")}
            features.append(ft)
            anchors.append({
                "anchor_id": f"fim_{lid}_{cat}", "source": "nws_fim",
                "name": f"{props.get('name', g.get('name',''))} ({cat} inundation)",
                "state": g.get("state", ""), "lat": round(la, 5), "lon": round(lo, 5),
                "severity_class": sev, "radius_km": round(clamp(rad, 1, 25), 1),
                "event_type": f"fim_{cat}", "valid_time": g.get("valid_time", ""),
                "gauge_id": g.get("gauge_id", ""), "gauge_stage_ft": g.get("gauge_stage_ft", ""),
            })
    print(f"  NWS-FIM inundation polygons: {len(features)} "
          f"(from {len(gauge_anchors)} in-flood gauges)")
    return anchors, features


# --------------------------------------------------------------------------- #
# Assemble + write
# --------------------------------------------------------------------------- #

ANCHOR_COLS = ["anchor_id", "source", "name", "state", "lat", "lon",
               "severity_class", "radius_km", "event_type", "valid_time",
               "gauge_id", "gauge_stage_ft"]


def write_outputs(anchors, features, out_dir):
    with open(f"{out_dir}/flood_anchors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ANCHOR_COLS)
        w.writeheader()
        for a in anchors:
            w.writerow({k: a.get(k, "") for k in ANCHOR_COLS})
    fc = {"type": "FeatureCollection", "features": features}
    with open(f"{out_dir}/flood_map.geojson", "w", encoding="utf-8") as f:
        json.dump(fc, f)
    print(f"  wrote {out_dir}/flood_anchors.csv  ({len(anchors)} anchors)")
    print(f"  wrote {out_dir}/flood_map.geojson  ({len(features)} features)")


def main():
    ap = argparse.ArgumentParser(description="Fetch real-time eastern-US flood map.")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--gauge-min-class", type=int, default=1,
                    help="Min gauge flood class to include (1=action..4=major).")
    ap.add_argument("--no-gauges", action="store_true")
    ap.add_argument("--no-alerts", action="store_true")
    ap.add_argument("--fema", action="store_true",
                    help="Also add FEMA NFHL flood-prone zones as anchors.")
    ap.add_argument("--fema-per-seed", type=int, default=3)
    ap.add_argument("--fim", action="store_true",
                    help="Also add NWS-FIM (CatFIM) real inundation polygons "
                         "for gauges currently in flood.")
    # Pass an empty list to parse_args() so it ignores sys.argv when in IPython/Jupyter
    import sys
    args = ap.parse_args(args=[] if 'ipykernel' in sys.modules or 'IPython' in sys.modules else None)

    anchors, features, gauge_anchors = [], [], []
    if not args.no_alerts:
        a, f = fetch_alerts(US_ALERT_AREAS)   # CONUS + DC + PR (no AK/HI)
        anchors += a; features += f
    if not args.no_gauges or args.fim:
        gauge_anchors, gf = fetch_gauges(US_TILES, args.gauge_min_class)
        if not args.no_gauges:
            anchors += gauge_anchors; features += gf
    if args.fim:
        a, f = fetch_nws_fim(gauge_anchors)
        anchors += a; features += f
    if args.fema:
        a, f = fetch_fema_nfhl(SEED_CITIES, per_seed=args.fema_per_seed)
        anchors += a; features += f

    mode = "live"
    if not anchors:
        # Fallback 1: most elevated gauges regardless of flood threshold.
        print("No active flooding found -- falling back to most elevated gauges ...")
        a, f = fetch_gauges(US_TILES, min_class=1)
        anchors += a; features += f
        mode = "fallback_gauges"
    if not anchors:
        # Fallback 2: bundled chronic flood locations.
        print("Still nothing -- using built-in chronic flood locations.")
        for name, st, lat, lon, sev in FALLBACK:
            anchors.append({"anchor_id": f"fb_{name}", "source": "fallback",
                            "name": name, "state": st, "lat": lat, "lon": lon,
                            "severity_class": sev, "radius_km": 10.0,
                            "event_type": "chronic", "valid_time": "",
                            "gauge_id": "", "gauge_stage_ft": ""})
        mode = "fallback_builtin"

    write_outputs(anchors, features, args.out_dir)
    print(f"\nMode: {mode}. Total anchors: {len(anchors)}. "
          f"Next: python3 generate_reports.py --anchors {args.out_dir}/flood_anchors.csv")


if __name__ == "__main__":
    main()
