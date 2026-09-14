from pathlib import Path

from .config import QWEN_IMAGE_WORKFLOW
from .workflow_runner import WorkflowRunner


class CompositionGenerator:

    def __init__(self):
        self.runner = WorkflowRunner()

    def build_prompt(self, scene, characters):
        character_text = []

        for character in characters:
            character_text.append(
                f"""
Character: {character.get("name", character["id"])}
Appearance: {character.get("appearance", "")}
Clothing: {character.get("clothing", "")}
Visual anchor: {character.get("visual_anchor", "")}
"""
            )

        characters_block = "\n".join(character_text)

        return f"""
Create ONE SINGLE CINEMATIC FILM FRAME for this story scene.

IMPORTANT:
- One continuous frame only.
- One camera angle only.
- One shot only.
- No collage.
- No split screen.
- No storyboard.
- No multiple panels.
- No duplicated characters.
- No multiple exposures.
- No text, captions or subtitles.

SCENE:
{scene.get("description", "")}

ACTION:
{scene.get("action", "")}

CAMERA:
{scene.get("camera", "")}

LIGHTING:
{scene.get("lighting", "")}

CHARACTERS:
{characters_block}

Create the complete 16:9 cinematic composition at 1280x720.

Establish the environment, location, weather, lighting, camera composition,
character placement and overall visual storytelling.

The image will later be used as the base composition for a
reference-conditioned character identity pass.

Photorealistic high-budget feature-film cinematography.
Realistic human anatomy.
Realistic skin.
Realistic clothing.
Detailed environment.
Natural depth of field.
Physically believable lighting.
Cinematic color grading.
"""
    
    def generate(
        self,
        scene,
        characters,
        output_path,
        seed,
    ):
        output_path = Path(output_path)

        workflow = self.runner.load_workflow(
            QWEN_IMAGE_WORKFLOW
        )

        prompt = self.build_prompt(
            scene,
            characters,
        )

        # Prompt
        workflow["5"]["inputs"]["text"] = prompt

        # Production scene resolution
        workflow["7"]["inputs"]["width"] = 1280
        workflow["7"]["inputs"]["height"] = 720
        workflow["7"]["inputs"]["batch_size"] = 1

        # Deterministic scene seed
        workflow["8"]["inputs"]["seed"] = seed

        # Strong anti-collage negative prompt
        workflow["6"]["inputs"]["text"] = (
            "low quality, blurry, distorted anatomy, deformed hands, "
            "extra limbs, duplicate people, duplicated characters, "
            "multiple people when not requested, collage, split screen, "
            "storyboard, comic panels, multiple panels, multiple views, "
            "multiple camera angles, double exposure, mirrored image, "
            "text, subtitles, captions, watermark, logo"
        )

        workflow["10"]["inputs"]["filename_prefix"] = (
            "story_compositions/scene"
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
                "Qwen composition generation completed "
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
            "prompt": prompt,
        }