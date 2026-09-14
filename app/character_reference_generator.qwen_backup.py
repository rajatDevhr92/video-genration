import re
from pathlib import Path

from .config import OUTPUT_DIR, WORKFLOWS_DIR
from .workflow_runner import WorkflowRunner


class CharacterReferenceGenerator:

    WORKFLOW = (
        WORKFLOWS_DIR
        / "qwen_image_2512_api.json"
    )

    def __init__(self):
        self.runner = WorkflowRunner()

    @staticmethod
    def safe_id(value: str) -> str:
        value = value.lower().strip()
        value = re.sub(r"[^a-z0-9]+", "_", value)
        return value.strip("_") or "character"

    @staticmethod
    def build_prompt(character):
        return f"""
Create ONE SINGLE photorealistic cinematic character reference image.

Character:
{character["name"]}

Appearance:
{character.get("appearance", "")}

Clothing:
{character.get("clothing", "")}

Personality:
{character.get("personality", "")}

Visual anchor:
{character.get("visual_anchor", "")}

Requirements:

One single person only.
Full body.
Standing naturally.
Face clearly visible.
Neutral or subtle serious expression appropriate to the character.
Realistic human anatomy.
Realistic skin and facial detail.
Detailed clothing.
Accurate hair.
Accurate eyes.
Consistent body proportions.

Create a clean production character reference suitable for maintaining
the character's visual identity across multiple cinematic scenes.

Neutral dark cinematic studio background.
Soft cinematic studio lighting.
Photorealistic.
High-end feature-film production quality.

Do not create multiple views.
Do not create a collage.
Do not create a split screen.
Do not create a storyboard.
Do not duplicate the character.
Do not include text, labels, captions or subtitles.
"""

    def generate(
        self,
        job_id,
        character,
        seed,
    ):
        character_id = character["id"]

        safe_name = self.safe_id(
            character.get(
                "name",
                character_id,
            )
        )

        output_path = (
            OUTPUT_DIR
            / job_id
            / "references"
            / "characters"
            / f"{safe_name}_{character_id}.png"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        workflow = self.runner.load_workflow(
            self.WORKFLOW
        )

        # Qwen T2I workflow
        workflow["5"]["inputs"]["text"] = (
            self.build_prompt(character)
        )

        workflow["7"]["inputs"]["width"] = 1024
        workflow["7"]["inputs"]["height"] = 1024
        workflow["7"]["inputs"]["batch_size"] = 1

        workflow["8"]["inputs"]["seed"] = seed

        workflow["10"]["inputs"]["filename_prefix"] = (
            f"story/{job_id}/characters/{safe_name}"
        )

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
                f"No image generated for character "
                f"{character_id}"
            )

        self.runner.download_image(
            images[0],
            output_path,
        )

        return {
            "character_id": character_id,
            "character_name": character.get(
                "name"
            ),
            "prompt_id": prompt_id,
            "client_id": client_id,
            "path": str(output_path),
        }