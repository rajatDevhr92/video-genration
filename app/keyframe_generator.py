import shutil
import uuid
from pathlib import Path

from .config import (
    COMFY_DIR,
    KEYFRAME_DENOISE,
    QWEN_EDIT_WORKFLOW,
)
from .workflow_runner import WorkflowRunner


class KeyframeGenerator:

    def __init__(self):
        self.runner = WorkflowRunner()
        self.input_dir = COMFY_DIR / "input"

    def generate(
        self,
        scene_image,
        character_references,
        prompt,
        output_path,
        seed,
    ):
        scene_image = Path(scene_image)
        output_path = Path(output_path)

        if not scene_image.exists():
            raise FileNotFoundError(
                f"Scene/base image not found: {scene_image}"
            )

        if not character_references:
            raise ValueError(
                "At least one character reference is required."
            )

        if len(character_references) > 3:
            raise ValueError(
                "Maximum of 3 character references supported."
            )

        workflow = self.runner.load_workflow(
            QWEN_EDIT_WORKFLOW
        )

        self.input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_files = []

        try:
            # --------------------------------------------------
            # Base image:
            #
            # Scene 1  -> T2I composition
            # Scene 2+ -> previous video's last frame
            # --------------------------------------------------

            base_name = (
                f"base_"
                f"{uuid.uuid4().hex[:12]}"
                f"{scene_image.suffix.lower()}"
            )

            base_target = (
                self.input_dir / base_name
            )

            shutil.copy2(
                scene_image,
                base_target,
            )

            temporary_files.append(
                base_name
            )

            workflow["1000"]["inputs"]["image"] = (
                base_name
            )

            workflow["16"]["inputs"]["image"] = [
                "1000",
                0,
            ]

            # --------------------------------------------------
            # Character references
            # --------------------------------------------------

            for index, reference in enumerate(
                character_references,
                start=1,
            ):
                reference = Path(reference)

                if not reference.exists():
                    raise FileNotFoundError(
                        f"Character reference not found: "
                        f"{reference}"
                    )

                ref_name = (
                    f"character_"
                    f"{uuid.uuid4().hex[:12]}"
                    f"{reference.suffix.lower()}"
                )

                ref_target = (
                    self.input_dir / ref_name
                )

                shutil.copy2(
                    reference,
                    ref_target,
                )

                temporary_files.append(
                    ref_name
                )

                node_id = str(1001 + index)

                workflow[node_id] = {
                    "class_type": "LoadImage",
                    "inputs": {
                        "image": ref_name
                    }
                }

                input_name = (
                    f"image{index}"
                )

                workflow["9"]["inputs"][
                    input_name
                ] = [
                    node_id,
                    0,
                ]

                workflow["13"]["inputs"][
                    input_name
                ] = [
                    node_id,
                    0,
                ]

            # Remove any unused optional references.
            for index in range(
                len(character_references) + 1,
                4,
            ):
                input_name = f"image{index}"

                workflow["9"]["inputs"].pop(
                    input_name,
                    None,
                )

                workflow["13"]["inputs"].pop(
                    input_name,
                    None,
                )

            # --------------------------------------------------
            # Identity + continuity
            # --------------------------------------------------

            workflow["9"]["inputs"]["prompt"] = (
                "Preserve the exact identity, facial structure, "
                "hairstyle, body proportions and clothing of "
                "all provided character reference images. "
                "Do not redesign, replace, duplicate or "
                "combine characters."
            )

            workflow["13"]["inputs"]["prompt"] = prompt

            # --------------------------------------------------
            # Sampling
            # --------------------------------------------------

            workflow["15"]["inputs"]["seed"] = seed
            workflow["15"]["inputs"]["steps"] = 40
            workflow["15"]["inputs"]["cfg"] = 4.0
            workflow["15"]["inputs"]["sampler_name"] = "euler"
            workflow["15"]["inputs"]["scheduler"] = "simple"
            workflow["15"]["inputs"]["denoise"] = (
                KEYFRAME_DENOISE
            )

            # --------------------------------------------------
            # Run
            # --------------------------------------------------

            prompt_id, client_id = (
                self.runner.submit(workflow)
            )

            result = self.runner.wait(
                prompt_id
            )

            images = self.runner.find_images(
                result
            )

            if not images:
                raise RuntimeError(
                    "Qwen Image Edit completed "
                    "without an image output."
                )

            self.runner.download_image(
                images[0],
                output_path,
            )

            return {
                "prompt_id": prompt_id,
                "client_id": client_id,
                "output": str(output_path),
                "base_image": str(scene_image),
                "character_references": [
                    str(x)
                    for x in character_references
                ],
                "denoise": KEYFRAME_DENOISE,
            }

        finally:
            for filename in temporary_files:
                (
                    self.input_dir / filename
                ).unlink(
                    missing_ok=True
                )