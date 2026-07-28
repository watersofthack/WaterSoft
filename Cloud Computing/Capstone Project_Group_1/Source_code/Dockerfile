# FloodOps multi-source flood-map API — self-refreshing.
# Serves the retrieved flood map on port 8000 AND re-fetches it every hour so
# the served data stays live without any host-side cron.
FROM python:3.11-slim

WORKDIR /app

# App + refresher + fetcher + a seed geojson (overwritten by the first refresh).
COPY serve_floodmap.py refresh_loop.py fetch_flood_map.py flood_map.geojson entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000
ENV REFRESH_SECONDS=3600

CMD ["/app/entrypoint.sh"]
