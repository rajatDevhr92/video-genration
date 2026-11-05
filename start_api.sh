#!/bin/bash

# Auto-install/verify CUDA 12 ONNX and cloning dependencies on startup if missing
python -c "import onnxruntime as ort; assert 'CUDAExecutionProvider' in ort.get_available_providers()" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "[STARTUP] Configuring CUDA 12 onnxruntime-gpu and cloning dependencies..."
    pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1
    pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
    pip install opencv-python-headless insightface python-multipart
fi

set -a
source /workspace/story-video-engine/.env.sh
set +a

cd /workspace/story-video-engine

exec python -m uvicorn api_server:app \
    --host 0.0.0.0 \
    --port 8000
