# FloodOps multi-source flood-map API — build, push, deploy

Self-contained build context. The service retrieves the real flood map from
several official sources and highlights where they DIFFER. Serves on port **8000**:
- `GET /flood_map.geojson` — the full combined map (all sources), as GeoJSON
- `GET /sources` — per-source feature counts + a simple cross-source discrepancy count

Replace `USERNAME` with your Docker Hub username (lowercase) in every command.

> IMPORTANT (Apple Silicon): your Mac is arm64 but the Jetstream VM is x86_64,
> so ALWAYS build with `--platform linux/amd64` or it won't run on the server.

## On your Mac

```
cd "/Users/ehsankahrizi/Library/CloudStorage/OneDrive-TheUniversityofAlabama/General/WaterSoftHack 2026/floodops-cloud"
docker build --platform linux/amd64 -t USERNAME/floodops-cloud:latest .
docker run --rm -p 8000:8000 USERNAME/floodops-cloud:latest      # test at http://localhost:8000/sources
docker login
docker push USERNAME/floodops-cloud:latest
```

## On the Jetstream VM (ssh exouser@149.165.170.6)

First time only — install Docker:
```
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER      # then log out and back in
```

Pull and run (auto-restarts on reboot):
```
docker pull USERNAME/floodops-cloud:latest
docker run -d --name floodops-cloud --restart unless-stopped -p 8000:8000 USERNAME/floodops-cloud:latest
docker ps
```

## Reach it

- Quick/private (SSH tunnel from your Mac):
  `ssh -L 8000:localhost:8000 exouser@149.165.170.6` → open http://localhost:8000/sources
- Public: open port 8000 in the Jetstream security group → http://149.165.170.6:8000/sources

## Update cycle (after changing files)

```
# Mac
docker build --platform linux/amd64 -t USERNAME/floodops-cloud:latest .
docker push USERNAME/floodops-cloud:latest
# Jetstream
docker pull USERNAME/floodops-cloud:latest
docker rm -f floodops-cloud
docker run -d --name floodops-cloud --restart unless-stopped -p 8000:8000 USERNAME/floodops-cloud:latest
```

## Refreshing the flood data

The image ships the currently retrieved `flood_map.geojson`. To update it, re-run
`1. fetch_flood_map.py` (in ../iOS_flood/synthetic_data), copy the new
`flood_map.geojson` here, then rebuild + push + pull. (Next step with Sunil:
run the fetch on an hourly schedule so the map stays live.)
