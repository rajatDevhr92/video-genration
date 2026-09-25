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
    genital_references=None,          # list of 0–2 paths
):
    scene_image = Path(scene_image)
    output_path = Path(output_path)
    genital_references = genital_references or []

    if not scene_image.exists():
        raise FileNotFoundError(f"Scene/base image not found: {scene_image}")

    if not character_references:
        raise ValueError("At least one character reference is required.")

    if len(character_references) > 3:
        raise ValueError("Maximum of 3 character references supported.")

    if len(genital_references) > 2:
        raise ValueError("Maximum of 2 genital reference images supported.")

    workflow = self.runner.load_workflow(QWEN_EDIT_WORKFLOW)
    self.input_dir.mkdir(parents=True, exist_ok=True)
    temporary_files = []

    try:
        # --------------------------------------------------
        # 1. Base image (Image 1)
        # --------------------------------------------------
        base_name = f"base_{uuid.uuid4().hex[:12]}{scene_image.suffix.lower()}"
        base_target = self.input_dir / base_name
        shutil.copy2(scene_image, base_target)
        temporary_files.append(base_name)

        workflow["1000"]["inputs"]["image"] = base_name
        workflow["16"]["inputs"]["image"] = ["1000", 0]
        workflow["9"]["inputs"]["image1"] = ["16", 0]
        workflow["13"]["inputs"]["image1"] = ["16", 0]

        # --------------------------------------------------
        # 2. Character references (Image 2, 3, 4)
        # --------------------------------------------------
        for index, reference in enumerate(character_references, start=1):
            reference = Path(reference)
            if not reference.exists():
                raise FileNotFoundError(f"Character reference not found: {reference}")

            ref_name = f"character_{uuid.uuid4().hex[:12]}{reference.suffix.lower()}"
            ref_target = self.input_dir / ref_name
            shutil.copy2(reference, ref_target)
            temporary_files.append(ref_name)

            node_id = str(1001 + index)
            workflow[node_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": ref_name}
            }

            ref_slot = index + 1          # Image 2, 3, 4
            input_name = f"image{ref_slot}"
            workflow["9"]["inputs"][input_name] = [node_id, 0]
            workflow["13"]["inputs"][input_name] = [node_id, 0]

        # --------------------------------------------------
        # 3. Genital references (Image 5 and Image 6)
        # --------------------------------------------------
        for index, genital_ref in enumerate(genital_references, start=1):
            genital_ref = Path(genital_ref)
            if not genital_ref.exists():
                raise FileNotFoundError(f"Genital reference not found: {genital_ref}")

            gen_name = f"genital_{uuid.uuid4().hex[:12]}{genital_ref.suffix.lower()}"
            gen_target = self.input_dir / gen_name
            shutil.copy2(genital_ref, gen_target)
            temporary_files.append(gen_name)

            node_id = str(1010 + index)   # use higher node IDs to avoid collision
            workflow[node_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": gen_name}
            }

            # Image 5 for first genital ref, Image 6 for second
            ref_slot = 4 + index
            input_name = f"image{ref_slot}"
            workflow["9"]["inputs"][input_name] = [node_id, 0]
            workflow["13"]["inputs"][input_name] = [node_id, 0]

        # Clean up any remaining unused slots (up to Image 6)
        max_used_slot = 1 + len(character_references) + len(genital_references)
        for ref_slot in range(max_used_slot + 1, 7):
            input_name = f"image{ref_slot}"
            workflow["9"]["inputs"].pop(input_name, None)
            workflow["13"]["inputs"].pop(input_name, None)

        # --------------------------------------------------
        # 4. Prompts
        # --------------------------------------------------
        # Negative prompt (Node 9) - keep short and focused
        workflow["9"]["inputs"]["prompt"] = (
            "bad anatomy, extra limbs, fused bodies, floating limbs, "
            "penis on woman, cock on female, dick on female body, "
            "vagina on man, floating penis, extra penis, "
            "genitals turning into objects, morphing genitals, "
            "wrong gender anatomy, swapped genitals, changing genital size, "
            "censorship, blur, mosaic, underwear on nude body, "
            "acne, pimples, skin blemishes, text, watermark"
        )

        # Positive prompt (Node 13)
        genital_lock = """
CRITICAL GENITAL IDENTITY LOCK (HIGHEST PRIORITY):
- Male character: exact realistic human penis, correct length, girth, color, vein detail, and glans shape. Penis must stay attached only to the male body at the correct anatomical position.
- Female character: exact realistic human vagina/pussy with correct labia shape, color, and placement between the thighs. NEVER attach any penis, dick, or cock to the female body.
- Keep exact genital size, shape, color, and proportions consistent with the reference images and previous frame.
- No floating penis, no extra penis, no genitals turning into objects, no gender swap of genitals, no size change during generation.
"""

        # If genital references are provided, reinforce them
        if genital_references:
            genital_lock += """
Image 5 and/or Image 6 are authoritative close-up references for the exact genital appearance.
Copy the exact shape, size, color, texture, and proportions from these genital reference images.
Do not invent, alter, or morph the genitals.
"""

        workflow["13"]["inputs"]["prompt"] = prompt + "\n\n" + genital_lock

        # --------------------------------------------------
        # 5. Sampling
        # --------------------------------------------------
        active_denoise = denoise if denoise is not None else KEYFRAME_DENOISE

        workflow["15"]["inputs"]["seed"] = seed
        workflow["15"]["inputs"]["steps"] = 40
        workflow["15"]["inputs"]["cfg"] = 4.0
        workflow["15"]["inputs"]["sampler_name"] = "euler"
        workflow["15"]["inputs"]["scheduler"] = "simple"
        workflow["15"]["inputs"]["denoise"] = float(active_denoise)

        # --------------------------------------------------
        # 6. Run
        # --------------------------------------------------
        prompt_id, client_id = self.runner.submit(workflow)
        result = self.runner.wait(prompt_id)
        images = self.runner.find_images(result)

        if not images:
            raise RuntimeError("Qwen Image Edit completed without an image output.")

        self.runner.download_image(images[0], output_path)

        return {
            "prompt_id": prompt_id,
            "client_id": client_id,
            "output": str(output_path),
            "base_image": str(scene_image),
            "character_references": [str(x) for x in character_references],
            "genital_references": [str(x) for x in genital_references],
            "denoise": active_denoise,
        }

    finally:
        for filename in temporary_files:
            (self.input_dir / filename).unlink(missing_ok=True)