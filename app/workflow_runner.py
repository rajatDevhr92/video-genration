import json
import time
import uuid
from pathlib import Path

import requests

from .config import COMFY_URL


class WorkflowRunner:

    def __init__(self, comfy_url=None):
        self.comfy_url = (
            comfy_url
            or COMFY_URL
        ).rstrip("/")

    def load_workflow(self, path):
        path = Path(path)

        with path.open(
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    def submit(self, workflow):
        client_id = str(uuid.uuid4())

        response = requests.post(
            f"{self.comfy_url}/prompt",
            json={
                "prompt": workflow,
                "client_id": client_id,
            },
            timeout=60,
        )

        if response.status_code != 200:
            raise RuntimeError(
                "ComfyUI rejected workflow:\n"
                f"HTTP {response.status_code}\n"
                f"{response.text}"
            )

        data = response.json()

        if "error" in data:
            raise RuntimeError(
                json.dumps(data, indent=2)
            )

        return data["prompt_id"], client_id

    def wait(
        self,
        prompt_id,
        timeout=1800,
        poll_seconds=2,
    ):
        started = time.time()

        while True:
            elapsed = time.time() - started

            if elapsed > timeout:
                raise TimeoutError(
                    f"ComfyUI job timed out: {prompt_id}"
                )

            response = requests.get(
                f"{self.comfy_url}/history/{prompt_id}",
                timeout=30,
            )

            response.raise_for_status()

            history = response.json()

            if prompt_id not in history:
                time.sleep(poll_seconds)
                continue

            result = history[prompt_id]

            status = result.get(
                "status",
                {}
            )

            if status.get("status_str") == "error":
                raise RuntimeError(
                    json.dumps(
                        result,
                        indent=2
                    )
                )

            if result.get("outputs"):
                return result

            time.sleep(poll_seconds)

    def download_image(
        self,
        image,
        output_path,
    ):
        response = requests.get(
            f"{self.comfy_url}/view",
            params={
                "filename": image["filename"],
                "subfolder": image.get(
                    "subfolder",
                    ""
                ),
                "type": image.get(
                    "type",
                    "output"
                ),
            },
            timeout=120,
        )

        response.raise_for_status()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_bytes(
            response.content
        )

        return output_path

    @staticmethod
    def find_images(result):
        images = []

        for node_id, node_output in (
            result.get("outputs", {}).items()
        ):
            for image in node_output.get(
                "images",
                []
            ):
                item = dict(image)
                item["node_id"] = node_id
                images.append(item)

        return images