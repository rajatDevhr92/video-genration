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
        denoise=None,
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

            workflow["1000"]["inputs"]["image"] = base_name

            workflow["16"]["inputs"]["image"] = [
                "1000",
                0,
            ]

            # CRITICAL: Image 1 in Qwen Edit Plus MUST ALWAYS BE base_image (Node 16)
            workflow["9"]["inputs"]["image1"] = [
                "16",
                0,
            ]
            workflow["13"]["inputs"]["image1"] = [
                "16",
                0,
            ]

            # --------------------------------------------------
            # Character references (mapped to Image 2, Image 3, Image 4)
            # --------------------------------------------------

            for index, reference in enumerate(
                character_references,
                start=1,
            ):
                reference = Path(reference)

                if not reference.exists():
                    raise FileNotFoundError(
                        f"Character reference not found: {reference}"
                    )

                ref_name = (
                    f"character_"
                    f"{uuid.uuid4().hex[:12]}"
                    f"{reference.suffix.lower()}"
                )

                ref_target = self.input_dir / ref_name

                shutil.copy2(
                    reference,
                    ref_target,
                )

                temporary_files.append(ref_name)

                node_id = str(1001 + index)

                workflow[node_id] = {
                    "class_type": "LoadImage",
                    "inputs": {
                        "image": ref_name
                    }
                }

                # Image slot index: Image 2 for char 1, Image 3 for char 2, Image 4 for char 3
                ref_slot = index + 1
                input_name = f"image{ref_slot}"

                workflow["9"]["inputs"][input_name] = [
                    node_id,
                    0,
                ]

                workflow["13"]["inputs"][input_name] = [
                    node_id,
                    0,
                ]

            # Remove any unused optional reference slots.
            for ref_slot in range(
                len(character_references) + 2,
                6,
            ):
                input_name = f"image{ref_slot}"

                workflow["9"]["inputs"].pop(
                    input_name,
                    None,
                )

                workflow["13"]["inputs"].pop(
                    input_name,
                    None,
                )

            # --------------------------------------------------
            # Identity + continuity prompt
            # --------------------------------------------------

            workflow["9"]["inputs"]["prompt"] = (
                "Preserve the exact facial structure, hairstyle, hair color, "
                "eye color, skin appearance, body build, and identity of each "
                "character from their respective reference image. "
                "Maintain background environment and scene layout from Image 1."
            )

            workflow["13"]["inputs"]["prompt"] = prompt

            # --------------------------------------------------
            # Sampling & Adaptive Denoise
            # --------------------------------------------------

            active_denoise = denoise if denoise is not None else KEYFRAME_DENOISE

            workflow["15"]["inputs"]["seed"] = seed
            workflow["15"]["inputs"]["steps"] = 40
            workflow["15"]["inputs"]["cfg"] = 4.0
            workflow["15"]["inputs"]["sampler_name"] = "euler"
            workflow["15"]["inputs"]["scheduler"] = "simple"
            workflow["15"]["inputs"]["denoise"] = float(active_denoise)

            # --------------------------------------------------
            # Run
            # --------------------------------------------------

            prompt_id, client_id = self.runner.submit(workflow)

            result = self.runner.wait(prompt_id)

            images = self.runner.find_images(result)

            if not images:
                raise RuntimeError(
                    "Qwen Image Edit completed without an image output."
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
                    str(x) for x in character_references
                ],
                "denoise": active_denoise,
            }

        finally:
            for filename in temporary_files:
                (self.input_dir / filename).unlink(missing_ok=True)