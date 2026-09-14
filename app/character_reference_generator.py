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
Photorealistic full-body editorial studio photograph of one adult
person standing alone.

SUBJECT:

{character["name"]}

Appearance:
{character.get("appearance", "")}

Outfit:
{character.get("clothing", "")}

Identity details:
{character.get("visual_anchor", "")}

Create a realistic photograph of this exact adult person.

The person must be the only human visible anywhere in the image.

The person is standing naturally in a relaxed three-quarter pose,
facing generally toward the camera.

Show the complete body from the top of the head to the bottoms of
both feet.

The entire body must fit comfortably inside the frame.

Both feet must be visible.

Both hands must be visible.

Use one consistent face and one consistent body.

The facial structure, hair, skin tone, eye color, body build and
distinctive physical features must match the supplied description.

The outfit must be exactly:

{character.get("clothing", "")}

Do not change the outfit.

Do not add clothing that is not specified.

Do not remove clothing that is specified.

Use a simple dark neutral studio background.

Professional cinematic photography.

Realistic human anatomy.

Realistic skin texture.

Realistic hair.

Realistic hands.

Realistic feet.

Natural proportions.

Natural perspective.

Soft directional studio lighting.

Subtle rim light.

Natural shadows.

High-end photorealistic photography.

The composition must be a single uninterrupted photograph with the
person centered in the frame and empty background surrounding them.

ONE ADULT PERSON.
ONE BODY.
ONE FACE.
ONE HEAD.
ONE OUTFIT.
ONE POSE.
ONE PHOTOGRAPH.
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
        # Character reference resolution
        # --------------------------------------------------

        workflow["7"]["inputs"]["width"] = 1024
        workflow["7"]["inputs"]["height"] = 1536
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
            "inset portrait, surrounding portrait, "
            "additional photograph, additional figure, "
            "multiple images, multiple photographs, "
            "multiple poses, multiple angles, "
            "side view, rear view, "
            "close-up, face close-up, "
            "cropped head, cropped hands, cropped feet, "
            "cropped body, "
            "collage, panel layout, grid layout, "
            "border, frame, split image, "
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
