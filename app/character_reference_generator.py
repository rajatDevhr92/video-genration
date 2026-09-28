import re
from pathlib import Path

from .config import OUTPUT_DIR, WORKFLOWS_DIR
from .workflow_runner import WorkflowRunner


class CharacterReferenceGenerator:

    WORKFLOW = (
        WORKFLOWS_DIR
        / "krea2_t2i_api.json"
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
Photorealistic studio character turnaround model sheet grid, showing 3 distinct full-body views of one adult person on a simple neutral background:

SUBJECT:
{character["name"]}

Appearance:
{character.get("appearance", "")}

Outfit:
{character.get("clothing", "")}

Identity details:
{character.get("visual_anchor", "")}

LAYOUT & VIEW SPECIFICATIONS:
- Left panel: Full frontal view, standing naturally and facing directly forward.
- Center panel: 3/4 side profile view, showing nose profile, jawline, side body structure.
- Right panel: Full rear/back view, showing hair length, back posture, and rear silhouette.

CRITICAL TURNAROUND RULES:
- The exact same face, hair, skin tone, eye color, body build, and outfit must be 100% consistent across all 3 view panels.
- Show the complete full body in each panel from top of head down to feet.
- Outfit must be exactly: {character.get("clothing", "")}
- Simple dark neutral studio background across all panels.
- Professional cinematic photorealistic studio photography with soft directional lighting.

ONE ADULT PERSON. 3-VIEW TURNAROUND MODEL SHEET GRID.
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

        prompt = self.build_prompt(
            character
        )

        # --------------------------------------------------
        # Krea 2 prompt
        # --------------------------------------------------

        workflow["5"]["inputs"]["text"] = prompt

        # --------------------------------------------------
        # Character reference resolution (Wide landscape for 3-panel turnaround grid)
        # --------------------------------------------------

        workflow["7"]["inputs"]["width"] = 1536
        workflow["7"]["inputs"]["height"] = 1024
        workflow["7"]["inputs"]["batch_size"] = 1

        # --------------------------------------------------
        # Deterministic character seed
        # --------------------------------------------------

        workflow["8"]["inputs"]["seed"] = seed

        # --------------------------------------------------
        # Krea 2 sampling
        # --------------------------------------------------

        workflow["8"]["inputs"]["steps"] = 6
        workflow["8"]["inputs"]["cfg"] = 1.0
        workflow["8"]["inputs"]["sampler_name"] = "euler"
        workflow["8"]["inputs"]["scheduler"] = "beta"
        workflow["8"]["inputs"]["denoise"] = 1.0

        # --------------------------------------------------
        # Negative prompt
        # --------------------------------------------------

        workflow["6"]["inputs"]["text"] = (
            "extra person, additional person, multiple people, "
            "duplicate person, duplicate body, duplicate face, "
            "extra head, extra face, extra body, "
            "extra arms, extra legs, extra hands, "
            "acne, pimples, skin blemishes, skin spots, rough skin texture, "
            "inset portrait, surrounding portrait, "
            "cropped head, cropped hands, cropped feet, "
            "cropped body, "
            "border, frame, "
            "text, labels, captions, logo, watermark"
        )

        # --------------------------------------------------
        # Output
        # --------------------------------------------------

        workflow["10"]["inputs"]["filename_prefix"] = (
            f"story/{job_id}/characters/{safe_name}"
        )

        print(
            f"[KREA2] Generating character reference "
            f"character={character_id} "
            f"seed={seed}"
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
                f"Krea 2 generated no image for "
                f"character {character_id}"
            )

        self.runner.download_image(
            images[0],
            output_path,
        )

        print(
            f"[KREA2] Character reference generated: "
            f"{output_path}"
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
