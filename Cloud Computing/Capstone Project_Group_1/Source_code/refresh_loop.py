#!/usr/bin/env python3
"""
Background refresher for the FloodOps container.

Runs `fetch_flood_map.py --fim --fema --out-dir /app` on a fixed interval
(default: every hour) so the served `flood_map.geojson` always reflects the
latest data from NWS-FIM / NWS alerts / NWPS gauges / FEMA NFHL.

Environment
-----------
    REFRESH_SECONDS   how often to re-fetch (default 3600 = 1 hour)

Started alongside serve_floodmap.py by entrypoint.sh.
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone

INTERVAL = int(os.environ.get("REFRESH_SECONDS", "3600"))
OUT_DIR = "/app"


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_once():
    print(f"[{now()}] refresh: fetching latest flood map ...", flush=True)
    try:
        r = subprocess.run(
            [sys.executable, "/app/fetch_flood_map.py",
             "--fim", "--fema", "--out-dir", OUT_DIR],
            timeout=300, capture_output=True, text=True,
        )
        if r.returncode == 0:
            print(f"[{now()}] refresh: OK", flush=True)
        else:
            print(f"[{now()}] refresh: FAILED (exit {r.returncode})", flush=True)
            print(r.stderr.strip()[-500:], flush=True)
    except subprocess.TimeoutExpired:
        print(f"[{now()}] refresh: TIMEOUT (>5 min); keeping previous map",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[{now()}] refresh: ERROR {e}", flush=True)


def main():
    print(f"refresher started; interval = {INTERVAL}s", flush=True)
    # First refresh runs immediately so the image ships a live snapshot.
    fetch_once()
    while True:
        time.sleep(INTERVAL)
        fetch_once()


if __name__ == "__main__":
    main()
