#!/bin/bash

set -a
source /workspace/story-video-engine/.env.sh
set +a

cd /workspace/story-video-engine

exec python -m uvicorn api_server:app \
    --host 0.0.0.0 \
    --port 8000
