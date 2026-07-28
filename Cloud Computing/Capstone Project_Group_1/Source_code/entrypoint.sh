#!/bin/sh
# Start the hourly refresher in the background, then run the web server in the
# foreground (container stays alive as long as the server does).
set -e
python -u /app/refresh_loop.py &
exec python -u /app/serve_floodmap.py
