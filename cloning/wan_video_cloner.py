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
import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

from app.config import (
    COMFY_DIR,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    WAN_I2V_WORKFLOW,
)
from app.composition_generator import CompositionGenerator
from app.workflow_runner import WorkflowRunner
from app.gpu_memory import print_gpu_memory

logger = logging.getLogger(__name__)


class WanVideoCloner:
    def __init__(self):
        self.runner = WorkflowRunner()
        self.input_dir = COMFY_DIR / "input"
        self.comp_generator = CompositionGenerator()

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

    def build_anime_prompt(self, character_prompts: List[str]) -> str:
        """Combines character specs into a rich anime cinematography prompt."""
        char_desc = ", ".join(p.strip().rstrip(".") for p in character_prompts if p.strip())
        if not char_desc:
            char_desc = "gorgeous anime woman with expressive eyes and stylized hair"

        return (
            f"masterpiece anime cinematography, {char_desc}, vibrant anime aesthetic, "
            f"clean cel-shaded linework, smooth skin, detailed lighting, sharp focus, 8k anime style, high animation quality"
        )

    def generate_anime_keyframe(self, prompt: str, output_path: Path, seed: int = 12345) -> Path:
        """Generates the high-resolution anime starting keyframe via Krea 2."""
        self.comp_generator.generate_simple(
            prompt=prompt,
            output_path=output_path,
            seed=seed,
        )
        return output_path

    def animate_with_wan(
        self,
        keyframe_path: Path,
        motion_prompt: str,
        output_video_path: Path,
        seed: int = 12345,
        length_frames: int = 81,
        fps: int = 16,
    ) -> Path:
        """Runs Wan 2.2 I2V on the generated keyframe to produce the high-res video."""
        keyframe_path = Path(keyframe_path)
        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        self.input_dir.mkdir(parents=True, exist_ok=True)
        temp_name = f"wan_cloning_{uuid.uuid4().hex[:12]}{keyframe_path.suffix.lower()}"
        temp_path = self.input_dir / temp_name
        shutil.copy2(keyframe_path, temp_path)

        workflow = self.runner.load_workflow(WAN_I2V_WORKFLOW)

        # Set input image and prompt
        workflow["1000"]["inputs"]["image"] = temp_name
        workflow["9"]["inputs"]["text"] = motion_prompt

        # Output dimensions and length (81 frames at 16 fps = ~5.06 sec)
        workflow["11"]["inputs"]["width"] = VIDEO_WIDTH
        workflow["11"]["inputs"]["height"] = VIDEO_HEIGHT
        workflow["11"]["inputs"]["length"] = length_frames
        workflow["15"]["inputs"]["fps"] = float(fps)

        workflow["12"]["inputs"]["noise_seed"] = seed
        workflow["13"]["inputs"]["noise_seed"] = seed
        workflow["16"]["inputs"]["filename_prefix"] = f"cloning/wan_{uuid.uuid4().hex[:12]}"

        logger.info(f"[WAN ANIME] Submitting Wan 2.2 I2V job (seed={seed}, frames={length_frames})...")
        print(f"[WAN ANIME] Submitting Wan 2.2 I2V job to ComfyUI (seed={seed}, length={length_frames} frames)...")
        prompt_id, client_id = self.runner.submit(workflow)
        print(f"[WAN ANIME] ComfyUI Prompt ID: {prompt_id} (Client: {client_id}). Waiting for GPU execution...")
        result = self.runner.wait(prompt_id, timeout=1800)
        print("[WAN ANIME] Wan 2.2 I2V execution completed successfully on GPU.")

        # Find produced video
        videos = []
        for node_id, node_output in result.get("outputs", {}).items():
            if "gifs" in node_output:
                videos.extend(node_output["gifs"])
            elif "videos" in node_output:
                videos.extend(node_output["videos"])

        if not videos:
            raise RuntimeError("Wan 2.2 I2V completed without video output.")

        # Download / copy to destination
        raw_video_path = output_video_path.parent / "temp_wan_raw.mp4"
        self.runner.download_video(videos[0], raw_video_path)
        return raw_video_path

    def clone_video_to_anime(
        self,
        source_video: Path,
        character_prompts: List[str],
        output_video: Path,
        job_dir: Path,
        seed: int = 42,
    ) -> Path:
        """End-to-end: Extract first frame -> Krea 2 Anime Keyframe -> Wan 2.2 I2V -> Audio Mux."""
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: Extract First Frame
        extracted_frame = job_dir / "source_first_frame.png"
        logger.info(f"[WAN ANIME] Extracting first frame from {source_video}")
        self.extract_first_frame(source_video, extracted_frame)

        # Step 2: Build Anime Prompt & Generate Keyframe
        anime_prompt = self.build_anime_prompt(character_prompts)
        anime_keyframe = job_dir / "anime_start_keyframe.png"
        logger.info(f"[WAN ANIME] Generating anime keyframe with prompt: {anime_prompt}")
        self.generate_anime_keyframe(anime_prompt, anime_keyframe, seed=seed)

        # Step 3: Animate with Wan 2.2
        motion_prompt = f"fluid anime motion, smooth animation, {anime_prompt}"
        print_gpu_memory("Before Wan 2.2 I2V ComfyUI execution")
        raw_wan_video = self.animate_with_wan(
            keyframe_path=anime_keyframe,
            motion_prompt=motion_prompt,
            output_video_path=output_video,
            seed=seed,
            length_frames=81,
            fps=VIDEO_FPS,
        )
        print_gpu_memory("After Wan 2.2 I2V ComfyUI execution")

        # Step 4: Re-attach Audio from original video
        logger.info(f"[WAN ANIME] Muxing original audio onto Wan video...")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(raw_wan_video),
            "-i", str(source_video),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            str(output_video),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            # Fallback if audio track is absent
            shutil.copy2(raw_wan_video, output_video)

        if raw_wan_video.exists():
            raw_wan_video.unlink(missing_ok=True)

        logger.info(f"[WAN ANIME] Complete! Saved to {output_video}")
        return output_video

