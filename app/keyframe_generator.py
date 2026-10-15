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
        genital_references=None,
        negative_prompt=None,
        anatomy_rules="",
    ):
        scene_image = Path(scene_image)
        output_path = Path(output_path)
        genital_references = genital_references or []

        if not scene_image.exists():
            raise FileNotFoundError(f"Scene/base image not found: {scene_image}")

        if not character_references and not genital_references:
            raise ValueError("At least one character or genital reference is required.")

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
            # 2. Reference Images Assembly (Slots: Image 2 and Image 3 ONLY)
            # TextEncodeQwenImageEditPlus in ComfyUI supports max 3 images (Image 1 base + Image 2 ref + Image 3 ref)
            # --------------------------------------------------
            all_references = []
            
            # Add character references first
            for ref in character_references:
                all_references.append(("character", Path(ref)))

            # Add genital references if space available (up to 2 total reference images)
            for gref in genital_references:
                if len(all_references) < 2:
                    all_references.append(("genital", Path(gref)))

            # Cap strictly at 2 reference images (for Image 2 and Image 3)
            all_references = all_references[:2]

            slot_descriptions = []
            for index, (ref_type, ref_path) in enumerate(all_references, start=1):
                if not ref_path.exists():
                    raise FileNotFoundError(f"{ref_type.capitalize()} reference not found: {ref_path}")

                ref_name = f"{ref_type}_{uuid.uuid4().hex[:12]}{ref_path.suffix.lower()}"
                ref_target = self.input_dir / ref_name
                shutil.copy2(ref_path, ref_target)
                temporary_files.append(ref_name)

                node_id = str(1001 + index)  # 1002 for Image 2, 1003 for Image 3
                workflow[node_id] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": ref_name}
                }

                ref_slot = index + 1  # Image 2, Image 3
                input_name = f"image{ref_slot}"
                workflow["9"]["inputs"][input_name] = [node_id, 0]
                workflow["13"]["inputs"][input_name] = [node_id, 0]

                if ref_type == "genital":
                    slot_descriptions.append(f"Image {ref_slot} (Genital Reference)")
                else:
                    slot_descriptions.append(f"Image {ref_slot} (Character Reference)")

            # Clean up all unused reference slots (image2, image3 if unused, and ALL higher slots like image4, image5, image6)
            for ref_slot in range(1 + len(all_references) + 1, 10):
                input_name = f"image{ref_slot}"
                workflow["9"]["inputs"].pop(input_name, None)
                workflow["13"]["inputs"].pop(input_name, None)

            # --------------------------------------------------
            # 3. Prompts
            # --------------------------------------------------
            workflow["9"]["inputs"]["prompt"] = negative_prompt or "bad anatomy, extra limbs, fused bodies, text, watermark"

            genital_lock = anatomy_rules or ""
            genital_slots = [s for s in slot_descriptions if "Genital" in s]
            if genital_slots:
                gen_str = " and ".join([s.split()[0] for s in genital_slots])
                genital_lock += f"""
- {gen_str} is the 2-angle anatomical detail grid reference (Frontal & 3/4 Profile) for genital appearance.
- Copy exact genital shape, skin tone, labia/penis structure, and anatomical details strictly from the matching angle panel of {gen_str}.
"""

            workflow["13"]["inputs"]["prompt"] = prompt + "\n\n" + genital_lock

            # --------------------------------------------------
            # 4. Sampling
            # --------------------------------------------------
            active_denoise = denoise if denoise is not None else KEYFRAME_DENOISE

            workflow["15"]["inputs"]["seed"] = seed
            workflow["15"]["inputs"]["steps"] = 40
            workflow["15"]["inputs"]["cfg"] = 4.0
            workflow["15"]["inputs"]["sampler_name"] = "euler"
            workflow["15"]["inputs"]["scheduler"] = "simple"
            workflow["15"]["inputs"]["denoise"] = float(active_denoise)

            # --------------------------------------------------
            # 5. Run
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

    def generate_stage_b(
        self,
        stage_a_image,
        genital_references,
        prompt,
        output_path,
        seed,
        denoise=0.30,
        negative_prompt=None,
        anatomy_rules="",
    ):
        """
        Stage B: Anatomy & Groin Refinement.
        Refines genitalia and contact areas using Stage A output as base,
        with low denoise to preserve face, body pose, and background.

        - stage_a_image: Path to Stage A output image (Image 1 base)
        - genital_references: List of genital close-up paths (max 1 or 2 for Image 2/3)
        - prompt: Refinement prompt
        - output_path: Output image path
        - seed: Random seed
        - denoise: Low denoise rate (default 0.30) to preserve identity while refining anatomy
        """
        stage_a_image = Path(stage_a_image)
        output_path = Path(output_path)
        genital_references = genital_references or []

        if not stage_a_image.exists():
            raise FileNotFoundError(f"Stage A image not found: {stage_a_image}")

        if not genital_references:
            raise ValueError("At least one genital reference is required for Stage B.")

        workflow = self.runner.load_workflow(QWEN_EDIT_WORKFLOW)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        temporary_files = []

        try:
            # --------------------------------------------------
            # 1. Base image (Image 1) = Stage A output
            # --------------------------------------------------
            base_name = f"base_b_{uuid.uuid4().hex[:12]}{stage_a_image.suffix.lower()}"
            base_target = self.input_dir / base_name
            shutil.copy2(stage_a_image, base_target)
            temporary_files.append(base_name)

            workflow["1000"]["inputs"]["image"] = base_name
            workflow["16"]["inputs"]["image"] = ["1000", 0]
            workflow["9"]["inputs"]["image1"] = ["16", 0]
            workflow["13"]["inputs"]["image1"] = ["16", 0]

            # --------------------------------------------------
            # 2. Reference Images Assembly (Slots: Image 2 and Image 3 ONLY)
            # --------------------------------------------------
            all_references = []

            # Add genital references (up to 2 total reference images)
            for gref in genital_references:
                if len(all_references) < 2:
                    all_references.append(("genital", Path(gref)))

            # Cap strictly at 2 reference images (for Image 2 and Image 3)
            all_references = all_references[:2]

            slot_descriptions = []
            slot_gender_locks = []
            for index, (ref_type, ref_path) in enumerate(all_references, start=1):
                if not ref_path.exists():
                    raise FileNotFoundError(f"Genital reference not found: {ref_path}")

                ref_name = f"{ref_type}_{uuid.uuid4().hex[:12]}{ref_path.suffix.lower()}"
                ref_target = self.input_dir / ref_name
                shutil.copy2(ref_path, ref_target)
                temporary_files.append(ref_name)

                node_id = str(1001 + index)  # 1002 for Image 2, 1003 for Image 3
                workflow[node_id] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": ref_name}
                }

                ref_slot = index + 1  # Image 2, Image 3
                input_name = f"image{ref_slot}"
                workflow["9"]["inputs"][input_name] = [node_id, 0]
                workflow["13"]["inputs"][input_name] = [node_id, 0]

                ref_fname = ref_path.name.lower()
                if "penis" in ref_fname:
                    slot_descriptions.append(f"Image {ref_slot} (Male Genital / Penis Reference)")
                    slot_gender_locks.append(f"  * Image {ref_slot} is STRICTLY for the MALE character's penis at his pelvis. DO NOT apply to female.")
                elif "pussy" in ref_fname or "vulva" in ref_fname:
                    slot_descriptions.append(f"Image {ref_slot} (Female Genital / Vulva Reference)")
                    slot_gender_locks.append(f"  * Image {ref_slot} is STRICTLY for the FEMALE character's pussy/vulva between thighs. DO NOT apply to male.")
                else:
                    slot_descriptions.append(f"Image {ref_slot} (Genital Reference)")

            # Clean up all unused reference slots (image2, image3 if unused, and ALL higher slots like image4, image5, image6)
            for ref_slot in range(1 + len(all_references) + 1, 10):
                input_name = f"image{ref_slot}"
                workflow["9"]["inputs"].pop(input_name, None)
                workflow["13"]["inputs"].pop(input_name, None)

            # --------------------------------------------------
            # 3. Prompts
            # --------------------------------------------------
            workflow["9"]["inputs"]["prompt"] = negative_prompt or "bad anatomy, extra limbs, fused bodies, text, watermark"

            genital_lock = anatomy_rules or ""
            gen_slots = [s for s in slot_descriptions if "Genital" in s]
            if gen_slots:
                gen_str = " and ".join([s.split()[0] for s in gen_slots])
                gender_rules_str = "\n".join(slot_gender_locks) if slot_gender_locks else ""
                genital_lock += f"""
- {gen_str} is the 2-angle anatomical detail grid reference (Frontal & 3/4 Profile) for genital appearance:
{gender_rules_str}
- Copy exact genital shape, skin tone, labia/penis structure, and anatomical details strictly from the matching angle panel of {gen_str}.
"""

            stage_b_contact_lock = """
=========================================================
CRITICAL CONTACT ZONE & STAGE B ANATOMY FOCUS
=========================================================
- Continuous solid skin between bodies across all contact areas.
- ABSOLUTELY NO black holes, dark voids, missing flesh, or shadow spots.
- Clean, sharp, realistic contact between male penis and female pussy.
- FEMALE ANATOMY: Only female pussy/vulva on female character. ABSOLUTELY NO penis on woman.
- MALE ANATOMY: Only male penis attached at pelvis on male character. ABSOLUTELY NO female vulva on man.
- Preserve face, hair, and background from Image 1 base; focus strictly on refining groin anatomy and contact alignment.
"""
            workflow["13"]["inputs"]["prompt"] = prompt + "\n\n" + genital_lock + "\n\n" + stage_b_contact_lock

            # --------------------------------------------------
            # 4. Sampling
            # --------------------------------------------------
            active_denoise = denoise if denoise is not None else 0.25

            workflow["15"]["inputs"]["seed"] = seed
            workflow["15"]["inputs"]["steps"] = 40
            workflow["15"]["inputs"]["cfg"] = 4.0
            workflow["15"]["inputs"]["sampler_name"] = "euler"
            workflow["15"]["inputs"]["scheduler"] = "simple"
            workflow["15"]["inputs"]["denoise"] = float(active_denoise)

            # --------------------------------------------------
            # 5. Run
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
                "base_image": str(stage_a_image),
                "genital_references": [str(x) for x in genital_references],
                "denoise": active_denoise,
            }

        finally:
            for filename in temporary_files:
                (self.input_dir / filename).unlink(missing_ok=True)