import os
import json
import time
import uuid
import shutil
import requests

COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188")

WORKFLOW_FILE = "/workspace/ComfyUI/wan2.2_t2v_14b.json"
OUTPUT_DIR = "/workspace/ComfyUI/output/video"


def submit_workflow(workflow):
    response = requests.post(
        f"{COMFY_URL}/prompt",
        json={
            "prompt": workflow,
            "client_id": str(uuid.uuid4())
        },
        timeout=60
    )

    response.raise_for_status()
    return response.json()


def wait_for_video(timeout=1800):
    start = time.time()

    while time.time() - start < timeout:

        files = []

        if os.path.exists(OUTPUT_DIR):
            for filename in os.listdir(OUTPUT_DIR):
                if filename.endswith(".mp4"):
                    path = os.path.join(OUTPUT_DIR, filename)
                    files.append((os.path.getmtime(path), path))

        if files:
            files.sort(reverse=True)

            modified_time, path = files[0]

     
