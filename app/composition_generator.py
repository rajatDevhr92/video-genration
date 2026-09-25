from pathlib import Path

from .config import WORKFLOWS_DIR
from .workflow_runner import WorkflowRunner


class CompositionGenerator:

    WORKFLOW = (
        WORKFLOWS_DIR
        / "krea2_t2i_api.json"
    )

    def __init__(self):
        self.runner = WorkflowRunner()

    def build_prompt(self, scene, characters):
        character_text = []

        for character in characters:
            character_text.append(
                f"""
CHARACTER:
Name: {character.get("name", character["id"])}

Appearance:
{character.get("appearance", "")}

Clothing:
{character.get("clothing", "")}

Visual anchor:
{character.get("visual_anchor", "")}
"""
            )

        characters_block = "\n".join(character_text)

        return f"""
Create ONE SINGLE photorealistic cinematic film frame
that visually depicts the exact story moment described below.

The image must look like a frame captured from a
high-budget feature film.

============================================================
STORY SCENE
============================================================

Scene description:
{scene.get("description", "")}

Action happening in this exact moment:
{scene.get("action", "")}

Camera direction:
{scene.get("camera", "")}

Lighting:
{scene.get("lighting", "")}

============================================================
CHARACTERS IN THE STORY
============================================================

{characters_block}

============================================================
STORY VISUALIZATION
============================================================

Translate the written story into a believable physical scene.

Show the characters performing the exact action described
in the story.

The environment must logically match the scene description.

Character placement must make physical and narrative sense.

If the story describes a character entering, exiting, sitting,
standing, walking, looking, reaching, holding something,
speaking to someone, reacting emotionally, or interacting
with the environment, visually depict that exact action.

Do not invent unrelated events.

Do not add unrelated characters.

Do not add unnecessary objects that are not appropriate
for the story moment.

The image should communicate the story moment visually
even without dialogue or text.

============================================================
CHARACTER CONSISTENCY
============================================================

Use the supplied character information as the identity
authority.

Preserve for every character:

- facial structure
- age
- gender
- hairstyle
- hair color
- eye color
- skin appearance
- body proportions
- clothing
- distinctive visual features
- visual anchor

Characters must remain visually believable and consistent.

============================================================
ANATOMY & GENDER LOCK (CRITICAL)
============================================================

- Every character must keep correct biological sex anatomy.
- Male characters must have a realistic human penis attached only to the male body.
- Female characters must have a realistic human vagina only.
- NEVER attach a penis to a female character.
- NEVER attach a vagina to a male character.
- Do not create floating genitals or extra genitals.
- Keep body proportions, genital size, and shape consistent with the character description.
- Realistic adult human anatomy only. No object morphing of body parts.

============================================================
ENVIRONMENT
============================================================

Build the environment directly from the story.

The location, architecture, furniture, props, weather,
time of day and surrounding atmosphere must logically
match the described scene.

Maintain realistic spatial relationships between:

- characters
- objects
- architecture
- foreground
- background

============================================================
CAMERA
============================================================

Follow the requested camera direction:

{scene.get("camera", "")}

Use ONE physical camera.

ONE viewpoint.

ONE continuous photographic frame.

No alternate views.

No second camera.

No camera collage.

============================================================
LIGHTING
============================================================

Follow the story lighting:

{scene.get("lighting", "")}

Lighting must be physically believable and consistent
with the environment and time of day.

============================================================
VISUAL STYLE
============================================================

Photorealistic human characters.

Realistic anatomy.

Realistic adult human genitals when the scene requires nudity. Correct gender anatomy only.

Realistic skin texture.

Realistic hair.

Realistic clothing materials.

Realistic environments.

Natural perspective.

Natural depth of field.

Physically believable lighting.

Subtle cinematic color grading.

High-end feature-film cinematography.

Detailed but natural image.

The result should look like a real frame from a
professionally produced live-action film.

============================================================
STRICT OUTPUT RULES
============================================================

ONE image.

ONE continuous frame.

ONE moment in time.

ONE camera.

ONE viewpoint.

No character sheet.

No reference sheet.

No turnaround sheet.

No multiple views.

No split screen.

No collage.

No storyboard.

No comic panels.

No multiple panels.

No duplicate characters.

No duplicated faces.

No duplicated bodies.

No alternate versions of characters.

No mirror images.

No double exposure.

No picture-in-picture.

No text.

No captions.

No subtitles.

No labels.

No watermark.

No logo.

Generate the exact cinematic story moment.
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
            self.WORKFLOW
        )

        prompt = self.build_prompt(
            scene,
            characters,
        )

        # --------------------------------------------------
        # Krea 2 prompt
        # --------------------------------------------------

        workflow["5"]["inputs"]["text"] = prompt

        # --------------------------------------------------
        # Production scene resolution
        # --------------------------------------------------

        workflow["7"]["inputs"]["width"] = 1280
        workflow["7"]["inputs"]["height"] = 720
        workflow["7"]["inputs"]["batch_size"] = 1

        # --------------------------------------------------
        # Deterministic scene seed
        # --------------------------------------------------

        workflow["8"]["inputs"]["seed"] = seed

        # --------------------------------------------------
        # Krea 2 recommended sampling
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
            "low quality, blurry, distorted anatomy, deformed hands, "
            "acne, pimples, skin blemishes, skin spots, rough skin, "
            "character merging, merged bodies, fused limbs, body clipping, "
            "extra limbs, duplicate people, duplicated characters, "
            "unwanted people, collage, split screen, storyboard, "
            "comic panels, multiple panels, multiple views, "
            "multiple camera angles, double exposure, mirrored image, "
            "censorship blur, mosaic, pixelation, censor bars, unwanted underwear, "
            "text, subtitles, captions, watermark, logo, "
            "penis on woman, cock on female, dick on female body, "
            "vagina on man, floating penis, extra penis, object instead of penis, "
            "genitals turning into objects, wrong gender anatomy, swapped genitals"
        )

        # --------------------------------------------------
        # Output
        # --------------------------------------------------

        workflow["10"]["inputs"]["filename_prefix"] = (
            "story_compositions/scene"
        )

        print(
            f"[KREA2] Generating composition "
            f"seed={seed} "
            f"output={output_path}"
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
                "Krea 2 composition generation "
                "completed without an image output."
            )

        self.runner.download_image(
            images[0],
            output_path,
        )

        print(
            f"[KREA2] Composition generated: "
            f"{output_path}"
        )

        return {
            "prompt_id": prompt_id,
            "client_id": client_id,
            "output": str(output_path),
            "prompt": prompt,
        }

    def generate_simple(
        self,
        prompt,
        output_path,
        seed=12345,
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        workflow = self.runner.load_workflow(self.WORKFLOW)

        workflow["5"]["inputs"]["text"] = prompt
        workflow["7"]["inputs"]["width"] = 1024
        workflow["7"]["inputs"]["height"] = 1024
        workflow["7"]["inputs"]["batch_size"] = 1
        workflow["8"]["inputs"]["seed"] = seed
        workflow["8"]["inputs"]["steps"] = 6
        workflow["8"]["inputs"]["cfg"] = 1.0
        workflow["8"]["inputs"]["sampler_name"] = "euler"
        workflow["8"]["inputs"]["scheduler"] = "beta"
        workflow["8"]["inputs"]["denoise"] = 1.0

        workflow["6"]["inputs"]["text"] = (
            "low quality, blurry, distorted anatomy, deformed hands, "
            "acne, pimples, skin blemishes, skin spots, text, watermark"
        )

        workflow["10"]["inputs"]["filename_prefix"] = "genital_refs/ref"

        print(f"[KREA2] Generating simple image: seed={seed}, output={output_path}")

        prompt_id, client_id = self.runner.submit(workflow)
        result = self.runner.wait(prompt_id)
        images = self.runner.find_images(result)

        if not images:
            raise RuntimeError("Krea 2 simple generation completed without an image output.")

        self.runner.download_image(images[0], output_path)

        return {
            "prompt_id": prompt_id,
            "client_id": client_id,
            "output": str(output_path),
            "prompt": prompt,
        }
