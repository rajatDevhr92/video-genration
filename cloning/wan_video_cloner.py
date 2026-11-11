"""
wan_video_cloner.py - Full-frame AI Video & Anime Cloner using Wan 2.2 I2V.

Workflow:
1. Extracts the initial keyframe from the source video using FFmpeg.
2. Restyles the initial keyframe into a high-resolution Anime / Photorealistic reference
   using Krea 2 (CompositionGenerator).
3. Animates the reference frame through Wan 2.2 I2V (dual expert high/low noise + Lightning LoRA)
   into native high-resolution video.
4. Muxes original video audio onto the final Wan animation via FFmpeg.
"""
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from app.config import (
    COMFY_DIR,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    WAN_I2V_WORKFLOW,
)
from app.composition_generator import CompositionGenerator
from app.keyframe_generator import KeyframeGenerator
from app.workflow_runner import WorkflowRunner
from app.gpu_memory import print_gpu_memory

logger = logging.getLogger(__name__)


class WanVideoCloner:
    def __init__(self):
        self.runner = WorkflowRunner()
        self.input_dir = COMFY_DIR / "input"
        self.comp_generator = CompositionGenerator()
        self.keyframe_generator = KeyframeGenerator()

    def probe_video(self, video_path: Path) -> Dict[str, Any]:
        """Auto-detects video resolution, framerate, duration, and audio track via ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            str(video_path)
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            info = json.loads(res.stdout)
            video_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
            audio_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)

            width = int(video_stream.get("width", 1280)) if video_stream else 1280
            height = int(video_stream.get("height", 720)) if video_stream else 720

            fps = 30.0
            if video_stream and "r_frame_rate" in video_stream:
                parts = video_stream["r_frame_rate"].split("/")
                if len(parts) == 2 and float(parts[1]) != 0:
                    fps = float(parts[0]) / float(parts[1])
                elif len(parts) == 1:
                    fps = float(parts[0])

            duration = float(info.get("format", {}).get("duration", 5.0))
            return {
                "width": width,
                "height": height,
                "fps": round(fps, 2),
                "duration": duration,
                "has_audio": audio_stream is not None,
            }
        except Exception as e:
            logger.warning(f"[PROBE] Could not probe {video_path}: {e}. Falling back to 1280x720 @ 30 FPS")
            return {
                "width": 1280,
                "height": 720,
                "fps": 30.0,
                "duration": 5.0,
                "has_audio": True,
            }

    def get_optimal_wan_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        """Maps input dimensions to optimal Wan 2.2 resolution preserving aspect ratio."""
        aspect = width / max(1, height)
        if aspect >= 1.5:  # Landscape 16:9 or wider (e.g. 1920x1080)
            return 1280, 720
        elif aspect <= 0.65:  # Vertical 9:16 (phone video, 1080x1920)
            return 720, 1280
        elif 0.9 <= aspect <= 1.1:  # Square 1:1
            return 960, 960
        else:
            # Custom aspect ratio: maintain proportional area near 1280x720
            target_area = 1280 * 720
            w = int(round((target_area * aspect) ** 0.5 / 16)) * 16
            h = int(round((target_area / aspect) ** 0.5 / 16)) * 16
            return max(512, w), max(512, h)

    def extract_first_frame(self, video_path: Path, output_image_path: Path) -> Path:
        """Extracts the first frame (time 00:00:00) of the source video."""
        output_image_path = Path(output_image_path)
        output_image_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-ss", "00:00:00.000",
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(output_image_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0 or not output_image_path.exists():
            raise RuntimeError(f"FFmpeg failed to extract first frame from {video_path}:\n{res.stderr}")
        return output_image_path

    def extract_best_keyframe(self, video_path: Path, output_image_path: Path) -> Path:
        """Extracts the starting scene keyframe from the source video."""
        return self.extract_first_frame(video_path, output_image_path)

    def build_solo_character_prompt(self, char_prompt: str, gender: str = "character", style: str = "anime") -> str:
        """
        Builds a dedicated solo portrait prompt for ONE single character.
        Explicitly prevents couple/grid/multi-person generation in Krea 2.
        """
        p_clean = char_prompt.strip().rstrip(".")
        style_clean = (style or "anime").lower().strip()

        if style_clean == "anime":
            return (
                f"masterpiece 2D Japanese anime solo portrait of 1 single {gender}, solo focus, centered face portrait, "
                f"{p_clean}, Kyoto Animation and Makoto Shinkai style, vibrant anime visual art, pure 2D anime illustration, "
                f"large expressive detailed anime eyes, gorgeous stylized anime hair, clean cel-shaded linework, "
                f"soft radiant anime studio lighting, sharp focus, 8k anime style, high animation quality, "
                f"no photorealism, no 3D, no real human skin texture, no other people, single character only"
            )

        return (
            f"photorealistic 8k cinematic 35mm film solo portrait of 1 single {gender}, centered face portrait, "
            f"{p_clean}, authentic human skin texture, natural skin pores, fine facial details, realistic human eyes, "
            f"soft volumetric natural lighting, shallow depth of field, sharp focus, raw photograph, "
            f"no other people, no extra faces, solo character only"
        )

    def build_character_prompt(self, character_prompts: List[str], style: str = "anime") -> str:
        """Combines character specs into a cinematography prompt for realistic or anime output."""
        char_desc = ", ".join(p.strip().rstrip(".") for p in character_prompts if p.strip())
        style_clean = (style or "anime").lower().strip()

        if style_clean == "anime":
            if not char_desc:
                char_desc = "gorgeous anime woman with expressive eyes and stylized hair"
            return (
                f"masterpiece 2D Japanese anime cinematography, {char_desc}, vibrant anime aesthetic, "
                f"clean cel-shaded linework, smooth skin, detailed lighting, sharp focus, 8k anime style, high animation quality"
            )

        # Default: Photorealistic
        if not char_desc:
            char_desc = "attractive adult person with authentic human facial features"

        return (
            f"photorealistic 8k cinematic 35mm film still of {char_desc}, authentic human skin texture, "
            f"natural skin pores, fine facial details, realistic human eyes, soft volumetric natural lighting, "
            f"depth of field, cinematic color grading, hyper-detailed, high production value, raw photograph"
        )

    def build_anime_prompt(self, character_prompts: List[str]) -> str:
        """Backwards-compatibility: builds anime prompt."""
        return self.build_character_prompt(character_prompts, style="anime")

    def generate_keyframe(
        self,
        prompt: str,
        output_path: Path,
        seed: int = 12345,
        negative_prompt: Optional[str] = None,
    ) -> Path:
        """Generates the high-resolution starting keyframe via Krea 2."""
        self.comp_generator.generate_simple(
            prompt=prompt,
            output_path=output_path,
            seed=seed,
            negative_prompt=negative_prompt,
        )
        return output_path

    def generate_anime_keyframe(self, prompt: str, output_path: Path, seed: int = 12345) -> Path:
        """Backwards-compatibility: generates starting keyframe."""
        return self.generate_keyframe(prompt, output_path, seed=seed)

    def animate_with_wan(
        self,
        keyframe_path: Path,
        motion_prompt: str,
        output_video_path: Path,
        seed: int = 12345,
        length_frames: int = 81,
        fps: int = 16,
        width: Optional[int] = None,
        height: Optional[int] = None,
        negative_prompt: Optional[str] = None,
    ) -> Path:
        """Runs Wan 2.2 I2V on the generated keyframe to produce the high-res video."""
        keyframe_path = Path(keyframe_path)
        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        self.input_dir.mkdir(parents=True, exist_ok=True)
        temp_name = f"wan_cloning_{uuid.uuid4().hex[:12]}{keyframe_path.suffix.lower()}"
        temp_path = self.input_dir / temp_name
        shutil.copy2(keyframe_path, temp_path)

        target_w = width or VIDEO_WIDTH
        target_h = height or VIDEO_HEIGHT

        try:
            workflow = self.runner.load_workflow(WAN_I2V_WORKFLOW)

            # Set input image and prompt
            workflow["1000"]["inputs"]["image"] = temp_name
            workflow["9"]["inputs"]["text"] = motion_prompt
            if negative_prompt:
                workflow["10"]["inputs"]["text"] = negative_prompt

            # Output dimensions and length (81 frames at 16 fps = ~5.06 sec)
            workflow["11"]["inputs"]["width"] = target_w
            workflow["11"]["inputs"]["height"] = target_h
            workflow["11"]["inputs"]["length"] = length_frames
            workflow["15"]["inputs"]["fps"] = float(fps)

            workflow["12"]["inputs"]["noise_seed"] = seed
            workflow["13"]["inputs"]["noise_seed"] = seed
            workflow["16"]["inputs"]["filename_prefix"] = f"cloning/wan_{uuid.uuid4().hex[:12]}"

            logger.info(f"[WAN CLONER] Submitting Wan 2.2 I2V job (seed={seed}, resolution={target_w}x{target_h}, frames={length_frames})...")
            print(f"[WAN CLONER] Submitting Wan 2.2 I2V job to ComfyUI ({target_w}x{target_h}, seed={seed}, length={length_frames} frames)...")
            prompt_id, client_id = self.runner.submit(workflow)
            print(f"[WAN CLONER] ComfyUI Prompt ID: {prompt_id} (Client: {client_id}). Waiting for GPU execution...")
            result = self.runner.wait(prompt_id, timeout=1800)
            print("[WAN CLONER] Wan 2.2 I2V execution completed successfully on GPU.")

            # Find produced video across outputs (videos, gifs, or animated images)
            videos = []
            for node_id, node_output in result.get("outputs", {}).items():
                if "gifs" in node_output:
                    videos.extend(node_output["gifs"])
                if "videos" in node_output:
                    videos.extend(node_output["videos"])
                if "images" in node_output:
                    animated = node_output.get("animated", [])
                    for idx, img in enumerate(node_output["images"]):
                        if isinstance(img, dict):
                            fname = img.get("filename", "").lower()
                            is_anim = idx < len(animated) and animated[idx] is True
                            if is_anim or fname.endswith((".mp4", ".mov", ".webm", ".mkv", ".gif")):
                                videos.append(img)

            if not videos:
                raise RuntimeError("Wan 2.2 I2V completed without video output.")

            # Download / copy to destination (compatible with any WorkflowRunner version)
            raw_video_path = output_video_path.parent / "temp_wan_raw.mp4"
            if hasattr(self.runner, "download_video"):
                self.runner.download_video(videos[0], raw_video_path)
            else:
                self.runner.download_image(videos[0], raw_video_path)
            return raw_video_path
        finally:
            temp_path.unlink(missing_ok=True)

    def clone_video(
        self,
        source_video: Path,
        character_prompts: List[str],
        output_video: Path,
        job_dir: Path,
        seed: int = 42,
        style: str = "anime",
        characters: Optional[List[Dict[str, Any]]] = None,
        denoise: float = 0.74,
    ) -> Path:
        """
        Auto-detecting Wan V2V Video Cloning Pipeline (Realistic Video -> Anime Video):
        1. Auto-detects video resolution, framerate, duration via ffprobe.
        2. Extracts the starting video scene (preserving room, bed, bodies, poses, and setting).
        3. Generates high-res character facial identity reference portraits via Krea 2.
        4. Re-synthesizes the scene with Qwen Image Edit Diffusion (Multi-Reference Flux Kontext),
           explicitly converting the realistic actors and environment into high-budget Japanese anime.
        5. Animates the anime scene in native high-resolution diffusion via Wan 2.2 14B Dual Experts.
        6. Up-samples to original video FPS and re-muxes original audio.
        """
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        style_clean = (style or "anime").lower().strip()

        # Step 1: Probe source video (auto-detect 1080p, 30 FPS, aspect ratio, audio)
        meta = self.probe_video(source_video)
        src_w, src_h = meta["width"], meta["height"]
        src_fps = meta["fps"]
        wan_w, wan_h = self.get_optimal_wan_dimensions(src_w, src_h)
        logger.info(f"[WAN CLONER] Auto-detected input: {src_w}x{src_h} @ {src_fps} FPS. Target Wan resolution: {wan_w}x{wan_h}")
        print(f"[WAN CLONER] Auto-detected input: {src_w}x{src_h} @ {src_fps} FPS. Target Wan resolution: {wan_w}x{wan_h}")

        # Step 2: Extract real scene keyframe (preserving room, bed, bodies, camera angle)
        extracted_frame = job_dir / "source_first_frame.png"
        logger.info(f"[WAN CLONER] Extracting starting scene frame from {source_video}")
        self.extract_best_keyframe(source_video, extracted_frame)

        # Step 3: Generate SEPARATE Individual Solo Character Identity Reference Photos
        generated_character_images = []
        female_ref_slot = None
        male_ref_slot = None
        female_desc = ""
        male_desc = ""

        for idx, char_prompt in enumerate(character_prompts):
            p_lower = char_prompt.lower()
            char_gender = None
            if characters and idx < len(characters) and characters[idx].get("gender"):
                char_gender = characters[idx]["gender"].lower()

            if char_gender == "female" or any(w in p_lower for w in ["woman", "female", "girl", "lady", "she", "her"]):
                gender_str = "female"
            elif char_gender == "male" or any(w in p_lower for w in ["man", "male", "guy", "boy", "he", "his"]):
                gender_str = "male"
            else:
                gender_str = f"character_{idx}"

            slot_num = idx + 2  # Image 2, Image 3
            if gender_str == "female" and female_ref_slot is None:
                female_ref_slot = slot_num
                female_desc = char_prompt
            elif gender_str == "male" and male_ref_slot is None:
                male_ref_slot = slot_num
                male_desc = char_prompt

            # Strict solo portrait prompt (NEVER combine multiple people into one image)
            solo_prompt = self.build_solo_character_prompt(char_prompt, gender=gender_str, style=style_clean)
            solo_face_path = job_dir / f"character_{idx}_{gender_str}_solo_face.png"

            logger.info(f"[WAN CLONER] Generating separate solo face for {gender_str} (Image {slot_num}) via Krea 2...")
            print(f"[WAN CLONER] Generating separate solo face for {gender_str} (Image {slot_num}) via Krea 2...")

            anime_negative_face = (
                "photorealistic, realistic photograph, 3d, 3d render, real human skin, "
                "natural skin pores, blemishes, raw photo, live action, camera photo, "
                "low quality, blurry, deformed anatomy, deformed hands, text, watermark"
            )
            char_seed = seed + idx * 7919
            self.generate_keyframe(
                solo_prompt,
                solo_face_path,
                seed=char_seed,
                negative_prompt=anime_negative_face if style_clean == "anime" else None,
            )
            generated_character_images.append(solo_face_path)

        # Keep character_identity_face.png for backwards compatibility
        if generated_character_images:
            shutil.copy2(generated_character_images[0], job_dir / "character_identity_face.png")

        # Step 4: Explicit Dual-Character Diffusion Identity Transfer via KeyframeGenerator (ZERO InSwapper)
        # Explicitly defines WHO is male and WHO is female in the scene and converts realistic actors to Anime
        styled_scene_keyframe = job_dir / "styled_scene_keyframe.png"
        logger.info(f"[WAN CLONER] Restyling scene keyframe to {style_clean} via Qwen Image Edit Diffusion (NO INSWAPPER)...")
        print(f"[WAN CLONER] Restyling scene keyframe to {style_clean} via Qwen Image Edit Diffusion (NO INSWAPPER)...")

        # Build highly specific instruction mapping each actor in the scene to their reference image
        f_slot_str = f"Image {female_ref_slot}" if female_ref_slot else "Image 2"
        m_slot_str = f"Image {male_ref_slot}" if male_ref_slot else "Image 3"

        if style_clean == "anime":
            edit_prompt = (
                f"Masterpiece 2D Japanese anime still frame, Kyoto Animation and Makoto Shinkai aesthetic. "
                f"Full 2D anime conversion: redraw and completely transform the real scene in Image 1 into high-budget flat cel-shaded 2D anime artwork.\n\n"
                f"MANDATORY DUAL-CHARACTER 2D ANIME TRANSFORMATION:\n"
                f"1. FEMALE ACTOR -> 2D ANIME CONVERSION ({f_slot_str}):\n"
                f"   - Identify the female actor in Image 1 (the woman kneeling, bending, or on top on the bed).\n"
                f"   - Completely transform her into a gorgeous 2D anime woman strictly matching {f_slot_str}.\n"
                f"   - Change her hair to the stylized anime hairstyle and color from {f_slot_str}.\n"
                f"   - Render her with clean cel-shaded anime linework, smooth radiant anime skin tone, large expressive anime eyes matching {f_slot_str}, and sharp anime contours.\n\n"
                f"2. MALE ACTOR -> 2D ANIME CONVERSION ({m_slot_str}):\n"
                f"   - Identify the male actor in Image 1 (the man sitting or lying on the bed).\n"
                f"   - Completely transform him into a handsome 2D anime man strictly matching {m_slot_str}.\n"
                f"   - Change his hair to the stylized anime haircut from {m_slot_str}.\n"
                f"   - Render him with clean cel-shaded anime linework, sharp anime jawline, smooth anime skin tone, and expressive anime eyes matching {m_slot_str}.\n\n"
                f"3. 2D ANIME ENVIRONMENT & POSE PRESERVATION:\n"
                f"   - Transform the entire room, bed, pillows, blankets, and background in Image 1 into beautifully rendered 2D anime background art with vibrant anime colors and luminous anime lighting.\n"
                f"   - Preserve 100% of the exact physical body positions, intimate interaction, angles, and camera composition from Image 1.\n"
                f"   - BOTH characters MUST be converted to 2D anime simultaneously. Clean cel-shaded anime art, pure 2D anime illustration, no photorealism, no real human skin texture, no photograph."
            )
            anime_qwen_negative = (
                "photorealistic, realistic, photograph, raw photo, real life, 3d, 3d render, cgi, "
                "real human skin, skin pores, camera noise, realistic lighting, live action, "
                "bad anatomy, extra limbs, fused bodies, text, watermark"
            )
            active_denoise = max(float(denoise), 0.82)
        else:
            edit_prompt = (
                f"Masterpiece 8k cinematic 35mm film still. Highly detailed photographic transformation of BOTH actors in Image 1.\n\n"
                f"MANDATORY DUAL-CHARACTER ROLE ASSIGNMENT:\n"
                f"1. FEMALE ACTOR REPLACEMENT ({f_slot_str}):\n"
                f"   - Identify the female actor in Image 1 (the woman kneeling, bending, or on top on the bed).\n"
                f"   - Completely transform her into the female character shown in {f_slot_str}.\n"
                f"   - Replace her hair with the styled neat hair from {f_slot_str}.\n"
                f"   - Change her skin tone across her entire body, back, shoulders, and face matching {f_slot_str}.\n"
                f"   - Transform her facial structure, eyes, and features to strictly match {f_slot_str}.\n\n"
                f"2. MALE ACTOR REPLACEMENT ({m_slot_str}):\n"
                f"   - Identify the male actor in Image 1 (the man sitting or lying on the bed).\n"
                f"   - Completely transform him into the male character shown in {m_slot_str}.\n"
                f"   - Replace his hair with the haircut from {m_slot_str}.\n"
                f"   - Change his skin tone across his entire body matching {m_slot_str}.\n"
                f"   - Transform his facial structure, jawline, and features to strictly match {m_slot_str}.\n\n"
                f"3. SCENE & ACTION PRESERVATION:\n"
                f"   - Preserve 100% of the original physical body positions, intimate interaction, bed, pillows, room background, lighting, and camera angle from Image 1.\n"
                f"   - BOTH the female actor AND the male actor MUST be transformed simultaneously.\n"
                f"   - Flawless authentic human skin texture, natural skin pores, realistic human eyes, cinematic volumetric lighting, raw photograph."
            )
            anime_qwen_negative = None
            active_denoise = float(denoise)

        self.keyframe_generator.generate(
            scene_image=extracted_frame,
            character_references=generated_character_images,
            prompt=edit_prompt,
            negative_prompt=anime_qwen_negative,
            output_path=styled_scene_keyframe,
            seed=seed,
            denoise=active_denoise,
        )

        # Step 5: Animate with Wan 2.2 starting from the diffusion styled scene frame
        if style_clean == "anime":
            motion_prompt = (
                f"masterpiece 2D Japanese anime animation, Kyoto Animation, Makoto Shinkai anime aesthetic, "
                f"smooth fluid hand-drawn anime movement, clean cel-shaded anime linework, gorgeous anime lighting, "
                f"vibrant anime color palette, pure 2D anime cinematography, high quality anime film"
            )
            wan_negative_prompt = (
                "photorealistic, realistic, photograph, raw photo, 3d, 3d render, live action, real human, "
                "skin pores, natural human skin texture, camera noise, real life footage, "
                "色调艳丽，过曝，静态，细节模糊不清，字幕，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
                "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合"
            )
        else:
            motion_prompt = f"photorealistic 8k cinematic film, continuous natural human movement, authentic physical motion, realistic action, high production cinematography"
            wan_negative_prompt = None

        print_gpu_memory("Before Wan 2.2 ComfyUI execution")
        raw_wan_video = self.animate_with_wan(
            keyframe_path=styled_scene_keyframe,
            motion_prompt=motion_prompt,
            output_video_path=output_video,
            seed=seed,
            length_frames=81,
            fps=16,
            width=wan_w,
            height=wan_h,
            negative_prompt=wan_negative_prompt,
        )
        print_gpu_memory("After Wan 2.2 ComfyUI execution")

        # Step 5: Smooth Framerate Resampling & Audio Muxing
        logger.info(f"[WAN CLONER] Resampling to original {src_fps} FPS and muxing audio...")
        temp_fps_video = job_dir / "temp_resampled_fps.mp4"

        # Resample to original FPS (e.g. 30 FPS) smoothly
        resample_cmd = [
            "ffmpeg", "-y",
            "-i", str(raw_wan_video),
            "-r", str(src_fps),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(temp_fps_video),
        ]
        res_fps = subprocess.run(resample_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        active_video = temp_fps_video if (res_fps.returncode == 0 and temp_fps_video.exists()) else raw_wan_video

        # Mux original audio onto output
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", str(active_video),
            "-i", str(source_video),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            str(output_video),
        ]
        res_mux = subprocess.run(mux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res_mux.returncode != 0:
            shutil.copy2(active_video, output_video)

        # Cleanup intermediate files
        raw_wan_video.unlink(missing_ok=True)
        temp_fps_video.unlink(missing_ok=True)

        logger.info(f"[WAN CLONER] Video cloning complete! Saved to {output_video}")
        return output_video

    def clone_video_to_anime(
        self,
        source_video: Path,
        character_prompts: List[str],
        output_video: Path,
        job_dir: Path,
        seed: int = 42,
    ) -> Path:
        """Backwards-compatibility wrapper for anime cloning."""
        return self.clone_video(
            source_video=source_video,
            character_prompts=character_prompts,
            output_video=output_video,
            job_dir=job_dir,
            seed=seed,
            style="anime",
        )

