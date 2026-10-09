import shutil
import time
import uuid
from pathlib import Path

import requests

from .config import (
    COMFY_DIR,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    WAN_I2V_WORKFLOW,
)
from .workflow_runner import WorkflowRunner


class VideoGenerator:

    def __init__(self):
        self.runner = WorkflowRunner()
        self.input_dir = COMFY_DIR / "input"

    def generate(
        self,
        keyframe,
        motion_prompt,
        output_path,
        seed,
        motion_lora_name=None,
        motion_lora_strength=0.65,
    ):
        keyframe = Path(keyframe)
        output_path = Path(output_path)

        if not keyframe.exists():
            raise FileNotFoundError(
                f"Keyframe not found: {keyframe}"
            )

        workflow = self.runner.load_workflow(
            WAN_I2V_WORKFLOW
        )

        self.input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_name = (
            f"wan_i2v_"
            f"{uuid.uuid4().hex[:12]}"
            f"{keyframe.suffix.lower()}"
        )

        temp_path = self.input_dir / temp_name

        shutil.copy2(
            keyframe,
            temp_path,
        )

        started = time.time()

        try:
            workflow["1000"]["inputs"]["image"] = temp_name

            workflow["9"]["inputs"]["text"] = motion_prompt

            if motion_lora_name:
                workflow["500"] = {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {
                        "model": ["5", 0],
                        "lora_name": motion_lora_name,
                        "strength_model": float(motion_lora_strength),
                    },
                }
                workflow["7"]["inputs"]["model"] = ["500", 0]

            workflow["11"]["inputs"]["width"] = VIDEO_WIDTH
            workflow["11"]["inputs"]["height"] = VIDEO_HEIGHT

            # 81 frames @ 16 FPS = ~5.06 seconds
            workflow["11"]["inputs"]["length"] = 81
            workflow["15"]["inputs"]["fps"] = float(VIDEO_FPS)

            workflow["12"]["inputs"]["noise_seed"] = seed
            workflow["13"]["inputs"]["noise_seed"] = seed

            workflow["16"]["inputs"]["filename_prefix"] = (
                f"story/videos/{uuid.uuid4().hex[:12]}"
            )

            prompt_id, client_id = self.runner.submit(
                workflow
            )

            result = self.runner.wait(
                prompt_id,
                timeout=1800,
            )

            videos = self.find_videos(result)

            if not videos:
                raise RuntimeError(
                    "Wan I2V completed without video output."
                )

            self.download_video(
                videos[0],
                output_path,
            )

            return {
                "prompt_id": prompt_id,
                "client_id": client_id,
                "output": str(output_path),
                "keyframe": str(keyframe),
                "seed": seed,
                "generation_time_seconds": (
                    time.time() - started
                ),
            }

        finally:
            temp_path.unlink(
                missing_ok=True
            )

    @staticmethod
    def find_videos(result):
        videos = []

        for node_id, node_output in (
            result.get("outputs", {}).items()
        ):
            # ComfyUI SaveVideo on this installation
            # returns MP4 files inside "images" and
            # marks them with animated=true.
            images = node_output.get("images", [])
            animated = node_output.get("animated", [])

            for index, item in enumerate(images):
                if not isinstance(item, dict):
                    continue

                filename = item.get("filename")

                if not filename:
                    continue

                # Only treat animated image outputs as videos.
                is_animated = (
                    index < len(animated)
                    and animated[index] is True
                )

                if not is_animated:
                    continue

                video = dict(item)
                video["node_id"] = node_id
                video["source"] = "images"
                videos.append(video)

            # Also support installations/nodes that return
            # a dedicated "videos" array.
            for item in node_output.get("videos", []):
                if not isinstance(item, dict):
                    continue

                if not item.get("filename"):
                    continue

                video = dict(item)
                video["node_id"] = node_id
                video["source"] = "videos"
                videos.append(video)

        return videos

    def download_video(
        self,
        video,
        output_path,
    ):
        response = requests.get(
            f"{self.runner.comfy_url}/view",
            params={
                "filename": video["filename"],
                "subfolder": video.get(
                    "subfolder",
                    ""
                ),
                "type": video.get(
                    "type",
                    "output"
                ),
            },
            timeout=300,
        )

        response.raise_for_status()

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_bytes(
            response.content
        )