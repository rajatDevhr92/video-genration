import json
import time
import uuid
from pathlib import Path

import requests


class ComfyClient:

    def __init__(
        self,
        base_url="http://127.0.0.1:8188",
        output_dir="/workspace/ComfyUI/output",
    ):
        self.base_url = base_url.rstrip("/")
        self.output_dir = Path(output_dir)

    def load_workflow(self, workflow_path):
        with open(workflow_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def submit(self, workflow):
        client_id = str(uuid.uuid4())

        response = requests.post(
            f"{self.base_url}/prompt",
            json={
                "prompt": workflow,
                "client_id": client_id,
            },
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        if "error" in data:
            raise RuntimeError(
                f"ComfyUI rejected workflow: {data}"
            )

        prompt_id = data.get("prompt_id")

        if not prompt_id:
            raise RuntimeError(
                f"No prompt_id returned by ComfyUI: {data}"
            )

        return prompt_id, client_id

    def get_history(self, prompt_id):
        response = requests.get(
            f"{self.base_url}/history/{prompt_id}",
            timeout=30,
        )

        response.raise_for_status()

        return response.json()

    def wait_for_completion(
        self,
        prompt_id,
        timeout=1800,
        poll_interval=2,
    ):
        started = time.time()

        while True:

            elapsed = time.time() - started

            if elapsed > timeout:
                raise TimeoutError(
                    f"ComfyUI generation timed out after "
                    f"{timeout} seconds. prompt_id={prompt_id}"
                )

            history = self.get_history(prompt_id)

            if prompt_id not in history:
                print(
                    f"Waiting for generation... "
                    f"{int(elapsed)}s",
                    flush=True,
                )

                time.sleep(poll_interval)
                continue

            job = history[prompt_id]

            status = job.get("status", {})

            if status.get("status_str") == "error":
                raise RuntimeError(
                    f"ComfyUI generation failed: "
                    f"{json.dumps(status, indent=2)}"
                )

            if status.get("completed") is False:
                print(
                    f"Generating... {int(elapsed)}s",
                    flush=True,
                )

                time.sleep(poll_interval)
                continue

            outputs = job.get("outputs", {})

            files = self._extract_output_files(outputs)

            if files:
                return files

            print(
                f"Generation finished but output "
                f"file not found yet... {int(elapsed)}s",
                flush=True,
            )

            time.sleep(poll_interval)

    def _extract_output_files(self, outputs):

        results = []

        def walk(value, node_id=None):

            if isinstance(value, dict):

                if "filename" in value:

                    filename = value.get("filename")
                    subfolder = value.get("subfolder", "")
                    file_type = value.get("type", "output")

                    if filename:
                        results.append({
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": file_type,
                            "node_id": node_id,
                        })

                for key, child in value.items():
                    walk(child, key)

            elif isinstance(value, list):

                for child in value:
                    walk(child, node_id)

        walk(outputs)

        resolved = []

        for item in results:

            if item["type"] not in ("output", "temp"):
                continue

            path = (
                self.output_dir
                / item["subfolder"]
                / item["filename"]
            ).resolve()

            if path.exists():
                item["path"] = str(path)
                resolved.append(item)

        return resolved