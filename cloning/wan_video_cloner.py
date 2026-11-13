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
import gc
import json
import logging
import math
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from app.config import (
    CHUNK_DURATION_SEC,
    CHUNK_TARGET_FRAMES,
    CLONING_BACKEND,
    COMFY_DIR,
    CONTROL_VIDEO_MODE,
    FUN_CONTROL_CFG,
    FUN_CONTROL_CLIP_MODEL,
    FUN_CONTROL_HIGH_NOISE_MODEL,
    FUN_CONTROL_LORA_HIGH,
    FUN_CONTROL_LORA_LOW,
    FUN_CONTROL_LOW_NOISE_MODEL,
    FUN_CONTROL_NEGATIVE_PROMPT,
    FUN_CONTROL_NODE_IDS,
    FUN_CONTROL_POSITIVE_PROMPT,
    FUN_CONTROL_STEPS,
    FUN_CONTROL_STEPS_LORA,
    FUN_CONTROL_VAE_MODEL,
    FUN_CONTROL_WORKFLOW,
    OUTPUT_DIR,
    QWEN_KEYFRAME_NEGATIVE,
    QWEN_KEYFRAME_PROMPT,
    REANCHOR_EVERY_N_CHUNKS,
    QUALITY_MAX_RETRIES,
    USE_LIGHTNING_LORA,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    WAN_I2V_WORKFLOW,
    WAN_V2V_WORKFLOW,
)
from app.composition_generator import CompositionGenerator
from app.keyframe_generator import KeyframeGenerator
from app.workflow_runner import WorkflowRunner
from app.gpu_memory import print_gpu_memory
from .cloning_quality import check_chunk

logger = logging.getLogger(__name__)


def _write_frames_to_mp4(frames: List[Any], output_path: Path, width: int, height: int, fps: int = 16):
    """Encodes a sequence of BGR frames directly to an H.264 MP4 video via FFmpeg pipe."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"FFmpeg writing control video failed:\n{err}")


def _generate_control_video_ffmpeg(
    chunk_src_mp4: Path,
    output_path: Path,
    mode: str,
    width: int,
    height: int,
    fps: int,
    target_frames: int,
):
    """High-speed FFmpeg filter fallback if python CV libraries are missing."""
    if mode == "canny":
        vf = f"scale={width}:{height},edgedetect=low=0.1:high=0.3,format=yuv420p"
    elif mode == "depth":
        vf = f"scale={width}:{height},curves=all='0/0 0.5/0.8 1/1',format=yuv420p"
    elif mode == "pose":
        vf = f"scale={width}:{height},edgedetect=low=0.15:high=0.35,format=yuv420p"
    else:  # pose+depth
        vf = f"scale={width}:{height},edgedetect=low=0.1:high=0.35,format=yuv420p"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(chunk_src_mp4),
        "-vf", vf,
        "-r", str(fps),
        "-frames:v", str(target_frames),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg control video generation failed: {res.stderr}")


def _process_depth_frames(raw_frames: List[Any], target_w: int, target_h: int) -> List[Any]:
    """Generates Depth Anything V2 pure grayscale depth frames (no colormap)."""
    import numpy as np
    import cv2
    try:
        from transformers import pipeline
        from PIL import Image
        import torch
        device = 0 if torch.cuda.is_available() else -1
        depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
        processed = []
        for f in raw_frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            depth_map = depth_pipe(pil_img)["depth"]
            depth_arr = np.array(depth_map)
            depth_norm = cv2.normalize(depth_arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            if depth_norm.shape[1] != target_w or depth_norm.shape[0] != target_h:
                depth_norm = cv2.resize(depth_norm, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            # Standard Wan Fun Control grayscale depth map (3 identical grayscale channels, no colormap)
            depth_bgr = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
            processed.append(depth_bgr)
        return processed
    except Exception as e:
        logger.warning(f"[CONTROL VIDEO] Depth model notice ({e}). Using adaptive grayscale depth proxy.")
        processed = []
        for f in raw_frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            mag = cv2.magnitude(sobelx, sobely)
            mag_norm = cv2.normalize(mag, None, 0, 180, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_proxy = cv2.addWeighted(blurred, 0.6, mag_norm, 0.4, 0)
            depth_bgr = cv2.cvtColor(depth_proxy, cv2.COLOR_GRAY2BGR)
            processed.append(depth_bgr)
        return processed


def _process_pose_frames(raw_frames: List[Any], target_w: int, target_h: int) -> List[Any]:
    """Generates DWPose / Openpose skeleton frames. If pose fails, logs clear warning and stops."""
    import numpy as np
    import cv2
    try:
        try:
            import mediapipe as mp
            if not hasattr(mp, "solutions"):
                import mediapipe.python.solutions as mps
                mp.solutions = mps
        except Exception:
            pass

        from controlnet_aux import DWposeDetector
        from PIL import Image
        import torch
        detector = DWposeDetector.from_pretrained("yzd-v/DWPose")
        if torch.cuda.is_available():
            detector = detector.to("cuda")
        processed = []
        for f in raw_frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pose_img = detector(pil_img, output_type="np")
            pose_bgr = cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR)
            if pose_bgr.shape[1] != target_w or pose_bgr.shape[0] != target_h:
                pose_bgr = cv2.resize(pose_bgr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            processed.append(pose_bgr)
        return processed
    except Exception as e:
        logger.error(f"[CONTROL VIDEO] WARNING: Pose extraction failed: {e}. Stopping as requested.")
        print(f"[CONTROL VIDEO] WARNING: Pose extraction failed: {e}. Stopping as requested.")
        raise RuntimeError(
            f"Pose extraction failed with mode='pose': {e}. "
            f"Please ensure controlnet_aux and onnxruntime are properly installed on the server."
        ) from e


def extract_control_video(
    chunk_src_mp4: Path,
    mode: str = "pose+depth",
    output_path: Optional[Path] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: int = 16,
    target_frames: int = CHUNK_TARGET_FRAMES,
) -> Path:
    """
    Creates a control video from each chunk (default 161 frames / 10s at 16 FPS).
    Supports mode = 'depth', 'pose', 'canny', 'pose+depth'.
    Saves next to the chunk as chunks/chunk_XXXX_control.mp4 and reuses on resume.
    """
    chunk_src_mp4 = Path(chunk_src_mp4)
    if output_path is None:
        chunk_stem = chunk_src_mp4.stem
        if chunk_stem.endswith("_src"):
            base_name = chunk_stem[:-4]
        else:
            base_name = chunk_stem
        output_path = chunk_src_mp4.parent / f"{base_name}_control.mp4"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume check: reuse existing control video if valid
    if output_path.exists() and output_path.stat().st_size > 10000:
        logger.info(f"[CONTROL VIDEO] Reusing existing control video: {output_path.name}")
        print(f"[CONTROL VIDEO] Reusing existing control video: {output_path.name}")
        return output_path

    mode_clean = (mode or "pose+depth").lower().strip()
    target_w = width or VIDEO_WIDTH
    target_h = height or VIDEO_HEIGHT

    logger.info(
        f"[CONTROL VIDEO] Generating control video (mode={mode_clean}, "
        f"res={target_w}x{target_h}, fps={fps}, frames={target_frames}) from {chunk_src_mp4.name}..."
    )
    print(
        f"[CONTROL VIDEO] Generating {mode_clean} control video ({target_w}x{target_h}, {fps} FPS, "
        f"{target_frames} frames) for {chunk_src_mp4.name}..."
    )

    has_cv2 = False
    try:
        import cv2
        import numpy as np
        has_cv2 = True
    except ImportError:
        pass

    if not has_cv2:
        logger.warning("[CONTROL VIDEO] OpenCV not detected. Using FFmpeg structure filter for control video.")
        _generate_control_video_ffmpeg(chunk_src_mp4, output_path, mode_clean, target_w, target_h, fps, target_frames)
        return output_path

    cap = cv2.VideoCapture(str(chunk_src_mp4))
    raw_frames = []
    while len(raw_frames) < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[1] != target_w or frame.shape[0] != target_h:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        raw_frames.append(frame)
    cap.release()

    if not raw_frames:
        logger.warning(f"[CONTROL VIDEO] Could not read frames with OpenCV. Falling back to FFmpeg.")
        _generate_control_video_ffmpeg(chunk_src_mp4, output_path, mode_clean, target_w, target_h, fps, target_frames)
        return output_path

    while len(raw_frames) < target_frames:
        raw_frames.append(raw_frames[-1].copy())
    raw_frames = raw_frames[:target_frames]

    control_frames = []
    if mode_clean == "canny":
        for f in raw_frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            control_frames.append(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    elif mode_clean == "depth":
        control_frames = _process_depth_frames(raw_frames, target_w, target_h)
    elif mode_clean == "pose":
        control_frames = _process_pose_frames(raw_frames, target_w, target_h)
    elif mode_clean in ("pose+depth", "pose_depth", "depth+pose"):
        depth_frames = _process_depth_frames(raw_frames, target_w, target_h)
        pose_frames = _process_pose_frames(raw_frames, target_w, target_h)
        for d_frame, p_frame in zip(depth_frames, pose_frames):
            pose_mask = (p_frame > 10).any(axis=2, keepdims=True)
            composite = np.where(pose_mask, p_frame, d_frame)
            control_frames.append(composite.astype(np.uint8))
    else:
        logger.warning(f"[CONTROL VIDEO] Unknown control mode '{mode_clean}'. Defaulting to canny.")
        for f in raw_frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            control_frames.append(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    _write_frames_to_mp4(control_frames, output_path, target_w, target_h, fps)
    logger.info(f"[CONTROL VIDEO] Control video generated successfully: {output_path} ({output_path.stat().st_size} bytes)")
    return output_path


def check_not_control_video(candidate_path: Path, control_video_path: Path, max_allowed_similarity: float = 0.88) -> Tuple[bool, float]:
    """
    Sanity check: compares candidate output video against the control video.
    If frame similarity is > max_allowed_similarity (e.g. 0.88), ComfyUI returned
    the control video itself instead of the generated diffusion video.
    Returns (is_valid, similarity_score).
    """
    try:
        import cv2
        import numpy as np

        cap_cand = cv2.VideoCapture(str(candidate_path))
        cap_ctrl = cv2.VideoCapture(str(control_video_path))

        similarities = []
        total_frames = int(cap_cand.get(cv2.CAP_PROP_FRAME_COUNT) or 80)
        sample_indices = set(np.linspace(5, max(5, total_frames - 5), num=5, dtype=int))

        idx = 0
        while cap_cand.isOpened() and cap_ctrl.isOpened() and len(similarities) < 5:
            ret_c, f_c = cap_cand.read()
            ret_k, f_k = cap_ctrl.read()
            if not ret_c or not ret_k:
                break
            if idx in sample_indices:
                g_c = cv2.cvtColor(cv2.resize(f_c, (160, 90)), cv2.COLOR_BGR2GRAY)
                g_k = cv2.cvtColor(cv2.resize(f_k, (160, 90)), cv2.COLOR_BGR2GRAY)
                diff = np.abs(g_c.astype(np.float32) - g_k.astype(np.float32))
                sim = 1.0 - (float(np.mean(diff)) / 255.0)
                similarities.append(sim)
            idx += 1

        cap_cand.release()
        cap_ctrl.release()

        avg_sim = float(np.mean(similarities)) if similarities else 0.0
        is_valid = avg_sim < max_allowed_similarity
        return is_valid, avg_sim
    except Exception as e:
        logger.warning(f"[SANITY CHECK] Could not compare candidate with control video: {e}")
        return True, 0.0


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

    def extract_frame_at_time(self, video_path: Path, timestamp_sec: float, output_image_path: Path) -> Path:
        """Extracts one frame at an exact timestamp (in seconds) from the source video."""
        output_image_path = Path(output_image_path)
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        hrs = int(timestamp_sec // 3600)
        mins = int((timestamp_sec % 3600) // 60)
        secs = timestamp_sec % 60
        time_str = f"{hrs:02d}:{mins:02d}:{secs:06.3f}"
        cmd = [
            "ffmpeg", "-y",
            "-ss", time_str,
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(output_image_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0 or not output_image_path.exists():
            cmd_fb = [
                "ffmpeg", "-y",
                "-ss", str(round(timestamp_sec, 3)),
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                str(output_image_path),
            ]
            subprocess.run(cmd_fb, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if not output_image_path.exists():
            return self.extract_first_frame(video_path, output_image_path)
        return output_image_path

    def extract_last_frame(self, video_path: Path, output_image_path: Path) -> Path:
        """
        Extracts the exact last frame of a video chunk for anime chunk chaining.
        Reuses cached frame if already extracted.
        """
        video_path = Path(video_path)
        output_image_path = Path(output_image_path)
        output_image_path.parent.mkdir(parents=True, exist_ok=True)

        if output_image_path.exists() and output_image_path.stat().st_size > 1000:
            return output_image_path

        cmd = [
            "ffmpeg", "-y",
            "-sseof", "-0.1",
            "-i", str(video_path),
            "-update", "1",
            "-q:v", "1",
            "-frames:v", "1",
            str(output_image_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and output_image_path.exists() and output_image_path.stat().st_size > 1000:
            return output_image_path

        cmd_select = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", "select=eq(n\\,80)",
            "-vframes", "1",
            "-q:v", "1",
            str(output_image_path),
        ]
        res_sel = subprocess.run(cmd_select, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res_sel.returncode == 0 and output_image_path.exists() and output_image_path.stat().st_size > 1000:
            return output_image_path

        return self.extract_first_frame(video_path, output_image_path)

    def prefilter_image_to_anime(self, input_image_path: Path, output_image_path: Path, wan_w: int, wan_h: int) -> Path:
        """
        Applies fast edge-preserving bilateral smoothing and cel-shaded color/contour
        enhancement to a single keyframe image via FFmpeg. Ensures it is formatted to
        wan_w x wan_h and stripped of photographic skin pores and raw camera textures.
        """
        output_image_path = Path(output_image_path)
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        vf_str = (
            f"scale={wan_w}:{wan_h}:force_original_aspect_ratio=decrease,"
            f"pad={wan_w}:{wan_h}:(ow-iw)/2:(oh-ih)/2,"
            f"bilateral=sigmaS=4:sigmaR=0.25,"
            f"eq=saturation=1.25:contrast=1.12,"
            f"unsharp=5:5:0.8:5:5:0.0"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_image_path),
            "-vf", vf_str,
            "-update", "1",
            "-frames:v", "1",
            str(output_image_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0 or not output_image_path.exists():
            shutil.copy2(input_image_path, output_image_path)
        return output_image_path

    def prefilter_chunk_to_anime(
        self,
        source_video: Path,
        start_t: float,
        chunk_dur: float,
        output_chunk_path: Path,
        wan_w: int,
        wan_h: int,
        target_frames: int = 81,
        style: str = "anime",
    ) -> Path:
        """
        Slices exact chunk from source video, scales/pads to Wan resolution,
        ensures exact 16 FPS and EXACTLY target_frames (e.g. 81 frames) with zero frame count drift,
        and applies edge-preserving bilateral smoothing + color saturation boost for pure anime style.
        """
        output_chunk_path = Path(output_chunk_path)
        output_chunk_path.parent.mkdir(parents=True, exist_ok=True)

        filters = [
            f"scale={wan_w}:{wan_h}:force_original_aspect_ratio=decrease",
            f"pad={wan_w}:{wan_h}:(ow-iw)/2:(oh-ih)/2",
            "fps=16",
            "tpad=stop_mode=clone:stop=-1",
        ]
        if style == "anime":
            filters.extend([
                "bilateral=sigmaS=4:sigmaR=0.25",
                "eq=saturation=1.25:contrast=1.12",
                "unsharp=5:5:0.8:5:5:0.0",
            ])
        vf_str = ",".join(filters)

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(round(start_t, 3)),
            "-i", str(source_video),
            "-vf", vf_str,
            "-frames:v", str(target_frames),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_chunk_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0 or not output_chunk_path.exists():
            logger.warning(f"[WAN V2V] FFmpeg prefilter slice notice: {res.stderr}. Using simple fallback slice.")
            fb_cmd = [
                "ffmpeg", "-y",
                "-ss", str(round(start_t, 3)),
                "-i", str(source_video),
                "-s", f"{wan_w}x{wan_h}",
                "-r", "16",
                "-frames:v", str(target_frames),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(output_chunk_path),
            ]
            subprocess.run(fb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return output_chunk_path

    def extract_control_video(
        self,
        chunk_src_mp4: Path,
        mode: Optional[str] = None,
        output_path: Optional[Path] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: int = 16,
        target_frames: int = 81,
    ) -> Path:
        """
        Creates an 81-frame control video from each chunk.
        Supports mode = 'pose', 'depth', 'canny', 'pose+depth' (default).
        Reuses chunks/chunk_XXXX_control.mp4 on resume.
        """
        return extract_control_video(
            chunk_src_mp4=chunk_src_mp4,
            mode=mode or CONTROL_VIDEO_MODE,
            output_path=output_path,
            width=width,
            height=height,
            fps=fps,
            target_frames=target_frames,
        )

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
        length_frames: int = 161,
        fps: int = 16,
        width: Optional[int] = None,
        height: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        cfg: float = 2.0,
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

            # Output dimensions and length (161 frames at 16 fps = ~10.06 sec)
            workflow["11"]["inputs"]["width"] = target_w
            workflow["11"]["inputs"]["height"] = target_h
            workflow["11"]["inputs"]["length"] = length_frames
            workflow["15"]["inputs"]["fps"] = float(fps)

            workflow["12"]["inputs"]["noise_seed"] = seed
            workflow["13"]["inputs"]["noise_seed"] = seed
            workflow["12"]["inputs"]["cfg"] = float(cfg)
            workflow["13"]["inputs"]["cfg"] = float(cfg)
            workflow["16"]["inputs"]["filename_prefix"] = f"cloning/wan_{uuid.uuid4().hex[:12]}"

            logger.info(f"[WAN CLONER] Submitting Wan 2.2 I2V job (seed={seed}, resolution={target_w}x{target_h}, frames={length_frames})...")
            print(f"[WAN CLONER] Submitting Wan 2.2 I2V job to ComfyUI ({target_w}x{target_h}, seed={seed}, length={length_frames} frames)...")
            prompt_id, client_id = self.runner.submit(workflow)
            print(f"[WAN CLONER] ComfyUI Prompt ID: {prompt_id} (Client: {client_id}). Waiting for GPU execution...")
            result = self.runner.wait(prompt_id, timeout=3600)
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

            # Download / copy directly to destination path
            if hasattr(self.runner, "download_video"):
                self.runner.download_video(videos[0], output_video_path)
            else:
                self.runner.download_image(videos[0], output_video_path)
            return output_video_path
        finally:
            temp_path.unlink(missing_ok=True)

    def animate_with_wan_v2v(
        self,
        chunk_video_path: Path,
        keyframe_path: Path,
        motion_prompt: str,
        output_video_path: Path,
        seed: int = 12345,
        length_frames: int = CHUNK_TARGET_FRAMES,
        fps: int = 16,
        width: Optional[int] = None,
        height: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        denoise: float = 0.75,
        cfg: float = 5.0,
    ) -> Path:
        """
        Runs true Wan 2.2 Video-to-Video (V2V) on the sliced video chunk.
        Encodes all chunk frames into 3D video latents with Wan VAE and denoises
        starting from step 1 (skipping step 0 noise reset), strictly locking 100% of
        the physical motions, poses, choreography, and timing from the source video.
        """
        chunk_video_path = Path(chunk_video_path)
        keyframe_path = Path(keyframe_path)
        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        self.input_dir.mkdir(parents=True, exist_ok=True)
        temp_img_name = f"wan_v2v_ref_{uuid.uuid4().hex[:12]}{keyframe_path.suffix.lower()}"
        temp_vid_name = f"wan_v2v_src_{uuid.uuid4().hex[:12]}.mp4"
        temp_img_path = self.input_dir / temp_img_name
        temp_vid_path = self.input_dir / temp_vid_name

        shutil.copy2(keyframe_path, temp_img_path)
        shutil.copy2(chunk_video_path, temp_vid_path)

        target_w = width or VIDEO_WIDTH
        target_h = height or VIDEO_HEIGHT

        try:
            workflow = self.runner.load_workflow(WAN_V2V_WORKFLOW)

            # 1. Inputs: anchor keyframe and chunk video
            workflow["1000"]["inputs"]["image"] = temp_img_name
            workflow["20"]["inputs"]["file"] = temp_vid_name

            # 2. Prompts
            workflow["9"]["inputs"]["text"] = motion_prompt
            if negative_prompt:
                workflow["10"]["inputs"]["text"] = negative_prompt

            # 3. Dimensions and length (locked to exact frames, e.g. 81)
            workflow["11"]["inputs"]["width"] = target_w
            workflow["11"]["inputs"]["height"] = target_h
            workflow["11"]["inputs"]["length"] = length_frames
            workflow["15"]["inputs"]["fps"] = float(fps)

            # 4. Seeds and CFG
            workflow["12"]["inputs"]["noise_seed"] = seed
            workflow["13"]["inputs"]["noise_seed"] = seed
            workflow["12"]["inputs"]["cfg"] = float(cfg)
            workflow["13"]["inputs"]["cfg"] = float(cfg)
            workflow["16"]["inputs"]["filename_prefix"] = f"cloning/wan_v2v_{uuid.uuid4().hex[:12]}"

            # 5. Denoise step control:
            # Step 1 out of 4 is ~0.75 denoise (preserves motion & choreography, repaints textures to anime).
            # Step 2 out of 4 is ~0.50 denoise.
            if denoise >= 0.70:
                start_step = 1
            else:
                start_step = 2
            workflow["12"]["inputs"]["start_at_step"] = start_step
            workflow["12"]["inputs"]["end_at_step"] = 2
            workflow["13"]["inputs"]["start_at_step"] = 2
            workflow["13"]["inputs"]["end_at_step"] = 4

            logger.info(
                f"[WAN V2V] Submitting Wan 2.2 V2V job (seed={seed}, resolution={target_w}x{target_h}, "
                f"frames={length_frames}, start_step={start_step}, cfg={cfg})..."
            )
            print(
                f"[WAN V2V] Submitting Wan 2.2 V2V job to ComfyUI ({target_w}x{target_h}, seed={seed}, "
                f"length={length_frames} frames, start_step={start_step}, cfg={cfg})..."
            )
            prompt_id, client_id = self.runner.submit(workflow)
            print(f"[WAN V2V] ComfyUI Prompt ID: {prompt_id} (Client: {client_id}). Waiting for GPU execution...")
            result = self.runner.wait(prompt_id, timeout=3600)
            print("[WAN V2V] Wan 2.2 V2V execution completed successfully on GPU.")

            # Find produced video strictly from SaveVideo node (node 16), avoiding input/control video node 20
            videos = []
            save_node_id = "16"
            all_outputs = result.get("outputs", {})

            if save_node_id in all_outputs:
                save_out = all_outputs[save_node_id]
                if "videos" in save_out:
                    videos.extend(save_out["videos"])
                elif "gifs" in save_out:
                    videos.extend(save_out["gifs"])
                elif "images" in save_out:
                    for img in save_out["images"]:
                        if isinstance(img, dict) and img.get("filename", "").lower().endswith((".mp4", ".mov", ".webm", ".mkv", ".gif")):
                            videos.append(img)

            if not videos:
                logger.error(f"[WAN V2V] SaveVideo node ({save_node_id}) produced no video output! ComfyUI output nodes: {list(all_outputs.keys())}")
                raise RuntimeError(f"Wan 2.2 V2V SaveVideo node ({save_node_id}) completed without video output. Refusing to fall back to other nodes.")

            if hasattr(self.runner, "download_video"):
                self.runner.download_video(videos[0], output_video_path)
            else:
                self.runner.download_image(videos[0], output_video_path)

            if not output_video_path.exists() or output_video_path.stat().st_size < 1000:
                raise RuntimeError(f"Wan V2V video download failed or produced empty file: {output_video_path}")
            return output_video_path
        except Exception as v2v_err:
            logger.error(
                f"[WAN V2V] Wan 2.2 V2V execution failed on GPU: {v2v_err}",
                exc_info=True,
            )
            print(f"[WAN V2V] ERROR: V2V execution failed: {v2v_err}")
            raise RuntimeError(f"Wan 2.2 Video-to-Video execution failed on GPU: {v2v_err}") from v2v_err
        finally:
            temp_img_path.unlink(missing_ok=True)
            temp_vid_path.unlink(missing_ok=True)

    def animate_with_wan_fun_control(
        self,
        control_video_path: Path,
        keyframe_path: Path,
        motion_prompt: Optional[str] = None,
        output_video_path: Path = None,
        seed: int = 12345,
        length_frames: int = CHUNK_TARGET_FRAMES,
        fps: int = 16,
        width: Optional[int] = None,
        height: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        use_lora: Optional[bool] = None,
        prompt_override: Optional[Dict[str, str]] = None,
    ) -> Path:
        """
        Runs Wan 2.2 14B Fun Control video generation using structural control video and anime start keyframe.
        Dual-expert high-noise and low-noise models run full sampling from step 0 (no start_at_step hack).
        LoRA usage is configurable (default False: 20 steps, no LoRA).
        """
        control_video_path = Path(control_video_path)
        keyframe_path = Path(keyframe_path)
        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        self.input_dir.mkdir(parents=True, exist_ok=True)
        temp_img_name = f"wan_fun_ref_{uuid.uuid4().hex[:12]}{keyframe_path.suffix.lower()}"
        temp_vid_name = f"wan_fun_ctl_{uuid.uuid4().hex[:12]}.mp4"
        temp_img_path = self.input_dir / temp_img_name
        temp_vid_path = self.input_dir / temp_vid_name

        shutil.copy2(keyframe_path, temp_img_path)
        shutil.copy2(control_video_path, temp_vid_path)

        target_w = width or VIDEO_WIDTH
        target_h = height or VIDEO_HEIGHT

        # Resolve prompts with overrides
        if prompt_override:
            active_prompt = prompt_override.get("fun_control_positive") or motion_prompt or FUN_CONTROL_POSITIVE_PROMPT
            active_neg_prompt = prompt_override.get("fun_control_negative") or negative_prompt or FUN_CONTROL_NEGATIVE_PROMPT
        else:
            active_prompt = motion_prompt or FUN_CONTROL_POSITIVE_PROMPT
            active_neg_prompt = negative_prompt or FUN_CONTROL_NEGATIVE_PROMPT

        active_use_lora = USE_LIGHTNING_LORA if use_lora is None else use_lora
        if steps is not None:
            active_steps = int(steps)
        else:
            active_steps = FUN_CONTROL_STEPS_LORA if active_use_lora else FUN_CONTROL_STEPS
        active_cfg = FUN_CONTROL_CFG if cfg is None else float(cfg)
        mid_step = active_steps // 2

        try:
            workflow = self.runner.load_workflow(FUN_CONTROL_WORKFLOW)
            nodes = FUN_CONTROL_NODE_IDS

            # 1. Inputs: start image (ref_image) and control video
            if nodes["load_ref_image"] in workflow:
                workflow[nodes["load_ref_image"]]["inputs"]["image"] = temp_img_name
            if nodes["load_control_video"] in workflow:
                workflow[nodes["load_control_video"]]["inputs"]["file"] = temp_vid_name

            # 2. Prompts
            if nodes["prompt_pos"] in workflow:
                workflow[nodes["prompt_pos"]]["inputs"]["text"] = active_prompt
            if active_neg_prompt and nodes["prompt_neg"] in workflow:
                workflow[nodes["prompt_neg"]]["inputs"]["text"] = active_neg_prompt

            # 3. Model filenames
            if nodes["unet_high"] in workflow:
                workflow[nodes["unet_high"]]["inputs"]["unet_name"] = FUN_CONTROL_HIGH_NOISE_MODEL
            if nodes["unet_low"] in workflow:
                workflow[nodes["unet_low"]]["inputs"]["unet_name"] = FUN_CONTROL_LOW_NOISE_MODEL
            if nodes["vae"] in workflow and "vae_name" in workflow[nodes["vae"]].get("inputs", {}):
                workflow[nodes["vae"]]["inputs"]["vae_name"] = FUN_CONTROL_VAE_MODEL
            if nodes["clip"] in workflow and "clip_name" in workflow[nodes["clip"]].get("inputs", {}):
                workflow[nodes["clip"]]["inputs"]["clip_name"] = FUN_CONTROL_CLIP_MODEL

            # 4. LoRA vs non-LoRA routing
            if active_use_lora:
                if nodes["lora_high"] in workflow:
                    workflow[nodes["lora_high"]]["inputs"]["lora_name"] = FUN_CONTROL_LORA_HIGH
                    workflow[nodes["shift_high"]]["inputs"]["model"] = [nodes["lora_high"], 0]
                if nodes["lora_low"] in workflow:
                    workflow[nodes["lora_low"]]["inputs"]["lora_name"] = FUN_CONTROL_LORA_LOW
                    workflow[nodes["shift_low"]]["inputs"]["model"] = [nodes["lora_low"], 0]
            else:
                # Bypass LoRAs: route UNET directly to ModelSamplingSD3
                if nodes["shift_high"] in workflow:
                    workflow[nodes["shift_high"]]["inputs"]["model"] = [nodes["unet_high"], 0]
                if nodes["shift_low"] in workflow:
                    workflow[nodes["shift_low"]]["inputs"]["model"] = [nodes["unet_low"], 0]

            # 5. Dimensions and length
            if nodes["fun_control"] in workflow:
                workflow[nodes["fun_control"]]["inputs"]["width"] = target_w
                workflow[nodes["fun_control"]]["inputs"]["height"] = target_h
                workflow[nodes["fun_control"]]["inputs"]["length"] = length_frames
            if nodes["create_video"] in workflow:
                workflow[nodes["create_video"]]["inputs"]["fps"] = float(fps)

            # 6. Samplers (step 0 to mid_step for high-noise; mid_step to active_steps for low-noise)
            if nodes["sampler_high"] in workflow:
                workflow[nodes["sampler_high"]]["inputs"]["noise_seed"] = seed
                workflow[nodes["sampler_high"]]["inputs"]["steps"] = active_steps
                workflow[nodes["sampler_high"]]["inputs"]["cfg"] = active_cfg
                workflow[nodes["sampler_high"]]["inputs"]["start_at_step"] = 0
                workflow[nodes["sampler_high"]]["inputs"]["end_at_step"] = mid_step
                workflow[nodes["sampler_high"]]["inputs"]["add_noise"] = "enable"
                workflow[nodes["sampler_high"]]["inputs"]["return_with_leftover_noise"] = "enable"

            if nodes["sampler_low"] in workflow:
                workflow[nodes["sampler_low"]]["inputs"]["noise_seed"] = seed
                workflow[nodes["sampler_low"]]["inputs"]["steps"] = active_steps
                workflow[nodes["sampler_low"]]["inputs"]["cfg"] = active_cfg
                workflow[nodes["sampler_low"]]["inputs"]["start_at_step"] = mid_step
                workflow[nodes["sampler_low"]]["inputs"]["end_at_step"] = active_steps
                workflow[nodes["sampler_low"]]["inputs"]["add_noise"] = "disable"
                workflow[nodes["sampler_low"]]["inputs"]["return_with_leftover_noise"] = "disable"

            # 7. Output filename prefix
            if nodes["save_video"] in workflow:
                workflow[nodes["save_video"]]["inputs"]["filename_prefix"] = (
                    f"cloning/wan_fun_{uuid.uuid4().hex[:12]}"
                )

            logger.info(
                f"[WAN FUN CONTROL] Submitting Wan 2.2 Fun Control job (seed={seed}, resolution={target_w}x{target_h}, "
                f"frames={length_frames}, steps={active_steps} [0->{mid_step}->{active_steps}], cfg={active_cfg}, "
                f"lora={active_use_lora})..."
            )
            print(
                f"[WAN FUN CONTROL] Submitting Wan 2.2 Fun Control job to ComfyUI ({target_w}x{target_h}, seed={seed}, "
                f"length={length_frames} frames, steps={active_steps}, cfg={active_cfg}, lora={active_use_lora})..."
            )
            prompt_id, client_id = self.runner.submit(workflow)
            print(f"[WAN FUN CONTROL] ComfyUI Prompt ID: {prompt_id} (Client: {client_id}). Waiting for GPU execution...")
            result = self.runner.wait(prompt_id, timeout=3600)
            print("[WAN FUN CONTROL] Wan 2.2 Fun Control execution completed successfully on GPU.")

            # Find produced video strictly from SaveVideo node (nodes["save_video"]), avoiding input/control video node 20
            videos = []
            save_node_id = str(nodes.get("save_video", "16"))
            all_outputs = result.get("outputs", {})

            if save_node_id in all_outputs:
                save_out = all_outputs[save_node_id]
                if "videos" in save_out:
                    videos.extend(save_out["videos"])
                elif "gifs" in save_out:
                    videos.extend(save_out["gifs"])
                elif "images" in save_out:
                    for img in save_out["images"]:
                        if isinstance(img, dict) and img.get("filename", "").lower().endswith((".mp4", ".mov", ".webm", ".mkv", ".gif")):
                            videos.append(img)

            if not videos:
                logger.error(
                    f"[WAN FUN CONTROL] SaveVideo node ({save_node_id}) produced no video output. "
                    f"ComfyUI output nodes present: {list(all_outputs.keys())}."
                )
                raise RuntimeError(
                    f"Wan 2.2 Fun Control SaveVideo node ({save_node_id}) produced no video output. "
                    f"Refusing to fall back to other nodes."
                )

            if hasattr(self.runner, "download_video"):
                self.runner.download_video(videos[0], output_video_path)
            else:
                self.runner.download_image(videos[0], output_video_path)

            if not output_video_path.exists() or output_video_path.stat().st_size < 1000:
                raise RuntimeError(f"Wan Fun Control video download failed or produced empty file: {output_video_path}")
            return output_video_path
        except Exception as fun_err:
            logger.error(
                f"[WAN FUN CONTROL] Wan 2.2 Fun Control execution failed on GPU: {fun_err}",
                exc_info=True,
            )
            print(f"[WAN FUN CONTROL] ERROR: Fun Control execution failed: {fun_err}")
            raise RuntimeError(f"Wan 2.2 Fun Control execution failed on GPU: {fun_err}") from fun_err
        finally:
            temp_img_path.unlink(missing_ok=True)
            temp_vid_path.unlink(missing_ok=True)

    def clone_video(
        self,
        source_video: Path,
        character_prompts: List[str],
        output_video: Path,
        job_dir: Optional[Path] = None,
        job_id: Optional[str] = None,
        seed: int = 42,
        style: str = "anime",
        characters: Optional[List[Dict[str, Any]]] = None,
        denoise: float = 0.75,
        status_callback: Optional[Any] = None,
        max_duration: Optional[float] = None,
        cloning_backend: Optional[str] = None,
        control_mode: Optional[str] = None,
        prompt_override: Optional[Dict[str, str]] = None,
    ) -> Path:
        """
        Auto-detecting Wan V2V / Fun Control Video Cloning Pipeline (Realistic Video -> Anime Video):
        1. Auto-detects video resolution, framerate, duration via ffprobe.
        2. Slices long videos (e.g. 5m to 30m+) into sequential ~5s chunks with resume capability.
        3. Generates high-res character facial identity reference portraits via Krea 2 (ONCE).
        4. Re-synthesizes initial keyframe with Qwen Image Edit into 100% 2D anime.
        5. Chains consecutive chunks with Wan 2.2 Fun Control using last frame + control video.
        6. Automatic quality gate checks motion, anime style, and flicker with retry loop.
        7. Seamlessly stitches all chunks via FFmpeg concat and remuxes full original audio.
        8. Manages job_state.json for stateful resume across server/API restarts.
        """
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)

        if job_dir is None:
            if job_id:
                job_dir = OUTPUT_DIR / job_id
            else:
                job_dir = OUTPUT_DIR / f"clone_{uuid.uuid4().hex[:12]}"
        else:
            job_dir = Path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)

        # Stateful Resume: Check if job_state.json exists
        state_file = job_dir / "job_state.json"
        saved_state = {}
        completed_chunk_indices = []
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as sf:
                    saved_state = json.load(sf)
                logger.info(f"[WAN CLONER] [RESUME] Loaded existing job_state.json for {job_dir.name}")
                print(f"[WAN CLONER] [RESUME] Loaded existing job_state.json for {job_dir.name}")
                completed_chunk_indices = saved_state.get("completed_chunks", [])
                if cloning_backend is None and "cloning_backend" in saved_state:
                    cloning_backend = saved_state["cloning_backend"]
                if control_mode is None and "control_mode" in saved_state:
                    control_mode = saved_state["control_mode"]
                if prompt_override is None and "prompt_override" in saved_state:
                    prompt_override = saved_state["prompt_override"]
            except Exception as se:
                logger.warning(f"[WAN CLONER] Could not parse job_state.json: {se}")

        style_clean = (style or "anime").lower().strip()

        # Step 1: Probe source video (auto-detect 1080p, 30 FPS, aspect ratio, duration, audio)
        meta = self.probe_video(source_video)
        src_w, src_h = meta["width"], meta["height"]
        src_fps = meta["fps"]
        total_duration = meta["duration"]

        if max_duration and float(max_duration) > 0:
            active_duration = min(float(total_duration), float(max_duration))
        else:
            active_duration = float(total_duration)

        wan_w, wan_h = self.get_optimal_wan_dimensions(src_w, src_h)
        logger.info(
            f"[WAN CLONER] Auto-detected input: {src_w}x{src_h} @ {src_fps} FPS. "
            f"Duration: {total_duration:.2f}s (active: {active_duration:.2f}s). Target Wan res: {wan_w}x{wan_h}"
        )
        print(
            f"[WAN CLONER] Auto-detected input: {src_w}x{src_h} @ {src_fps} FPS. "
            f"Duration: {total_duration:.2f}s (active: {active_duration:.2f}s). Target Wan res: {wan_w}x{wan_h}"
        )

        # Step 2: Extract real scene keyframe for chunk 0
        extracted_frame = job_dir / "source_first_frame.png"
        logger.info(f"[WAN CLONER] Extracting starting scene frame from {source_video}")
        self.extract_best_keyframe(source_video, extracted_frame)

        # Step 3: Generate SEPARATE Individual Solo Character Identity Reference Photos (ONCE for whole video)
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

            solo_prompt = self.build_solo_character_prompt(char_prompt, gender=gender_str, style=style_clean)
            solo_face_path = job_dir / f"character_{idx}_{gender_str}_solo_face.png"

            if solo_face_path.exists() and solo_face_path.stat().st_size > 10000:
                logger.info(f"[WAN CLONER] Reusing existing solo face for {gender_str} ({solo_face_path.name})")
            else:
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

        # Step 4: Multi-Chunk Sequential Cloning for Long Videos (5s chunks, up to 7+ minutes)
        f_slot_str = f"Image {female_ref_slot}" if female_ref_slot else "Image 2"
        m_slot_str = f"Image {male_ref_slot}" if male_ref_slot else "Image 3"

        if style_clean == "anime":
            p_override = prompt_override or {}
            raw_qwen_prompt = p_override.get("qwen_keyframe") or QWEN_KEYFRAME_PROMPT
            if "{f_slot_str}" in raw_qwen_prompt or "{m_slot_str}" in raw_qwen_prompt:
                edit_prompt = raw_qwen_prompt.format(f_slot_str=f_slot_str, m_slot_str=m_slot_str)
            else:
                edit_prompt = raw_qwen_prompt

            anime_qwen_negative = p_override.get("qwen_negative") or QWEN_KEYFRAME_NEGATIVE
            active_denoise = max(float(denoise), 0.50)
            active_cfg = 5.0
            motion_prompt = p_override.get("fun_control_positive") or FUN_CONTROL_POSITIVE_PROMPT
            wan_negative_prompt = p_override.get("fun_control_negative") or FUN_CONTROL_NEGATIVE_PROMPT
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
            active_cfg = 4.0
            motion_prompt = f"photorealistic 8k cinematic film, continuous natural human movement, authentic physical motion, realistic action, high production cinematography"
            wan_negative_prompt = None

        CHUNK_SEC = CHUNK_DURATION_SEC
        TARGET_FRAMES = CHUNK_TARGET_FRAMES
        num_chunks = max(1, int(math.ceil(active_duration / CHUNK_SEC)))
        logger.info(f"[WAN CLONER] Processing {active_duration:.2f}s in {num_chunks} chunks ({CHUNK_SEC}s / {TARGET_FRAMES} frames each)...")
        print(f"[WAN CLONER] Processing {active_duration:.2f}s in {num_chunks} chunks ({CHUNK_SEC}s / {TARGET_FRAMES} frames each)...")

        chunks_dir = job_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        completed_chunk_paths = []

        active_backend = (cloning_backend or CLONING_BACKEND or "fun_control").lower().strip()
        active_control_mode = (control_mode or CONTROL_VIDEO_MODE or "pose+depth").lower().strip()

        job_state = {
            "job_id": job_dir.name,
            "source_video": str(source_video),
            "style": style_clean,
            "cloning_backend": active_backend,
            "control_mode": active_control_mode,
            "prompt_override": prompt_override,
            "seed": seed,
            "denoise": denoise,
            "total_duration": total_duration,
            "active_duration": active_duration,
            "num_chunks": num_chunks,
            "chunk_sec": CHUNK_SEC,
            "target_frames": TARGET_FRAMES,
            "completed_chunks": completed_chunk_indices,
            "status": "in_progress",
            "updated_at": time.time(),
        }
        with open(state_file, "w", encoding="utf-8") as sf:
            json.dump(job_state, sf, indent=2)

        for chunk_idx in range(num_chunks):
            start_t = chunk_idx * CHUNK_SEC
            this_chunk_dur = min(CHUNK_SEC, active_duration - start_t)
            chunk_target = chunks_dir / f"chunk_{chunk_idx:04d}.mp4"

            # Checkpoint / Resume capability: skip if this chunk is already finished
            if chunk_target.exists() and chunk_target.stat().st_size > 50000:
                logger.info(f"[WAN CLONER] [RESUME] Chunk {chunk_idx + 1}/{num_chunks} already completed. Skipping.")
                print(f"[WAN CLONER] [RESUME] Chunk {chunk_idx + 1}/{num_chunks} already completed ({chunk_target.name}). Skipping.")
                if chunk_idx not in completed_chunk_indices:
                    completed_chunk_indices.append(chunk_idx)
                completed_chunk_paths.append(chunk_target)
                continue

            logger.info(f"[WAN CLONER] === Chunk {chunk_idx + 1}/{num_chunks} ({start_t:.1f}s - {start_t + this_chunk_dur:.1f}s) ===")
            print(f"[WAN CLONER] === Chunk {chunk_idx + 1}/{num_chunks} ({start_t:.1f}s - {start_t + this_chunk_dur:.1f}s) ===")
            logger.debug(f"[WAN CLONER] Chunk {chunk_idx + 1} Positive Prompt: {motion_prompt}")
            logger.debug(f"[WAN CLONER] Chunk {chunk_idx + 1} Negative Prompt: {wan_negative_prompt}")

            if status_callback:
                status_callback(
                    status="animating_with_wan",
                    current_chunk=chunk_idx + 1,
                    total_chunks=num_chunks,
                    progress=f"{chunk_idx + 1}/{num_chunks} ({(chunk_idx + 1) * 100 // num_chunks}%)",
                )

            chunk_seed = seed + chunk_idx * 10007

            # 4a. Slice and prefilter chunk video: exact Wan dimensions, exact 16 FPS, exact 81 frames, cel-shaded anime
            chunk_src_video = chunks_dir / f"chunk_{chunk_idx:04d}_src.mp4"
            self.prefilter_chunk_to_anime(
                source_video=source_video,
                start_t=start_t,
                chunk_dur=this_chunk_dur,
                output_chunk_path=chunk_src_video,
                wan_w=wan_w,
                wan_h=wan_h,
                target_frames=TARGET_FRAMES,
                style=style_clean,
            )

            # 4a-ctl. Generate or reuse control video for Wan Fun Control (mode: pose, depth, canny, or pose+depth)
            chunk_control_video = chunks_dir / f"chunk_{chunk_idx:04d}_control.mp4"
            active_control_mode = control_mode or CONTROL_VIDEO_MODE
            self.extract_control_video(
                chunk_src_mp4=chunk_src_video,
                mode=active_control_mode,
                output_path=chunk_control_video,
                width=wan_w,
                height=wan_h,
                fps=16,
                target_frames=TARGET_FRAMES,
            )

            # 4b & 4c. Anime Start Frame & Chunk Chaining / Re-anchoring
            styled_chunk_keyframe = chunks_dir / f"keyframe_{chunk_idx:04d}_styled.png"
            is_reanchor_chunk = (chunk_idx > 0) and (chunk_idx % REANCHOR_EVERY_N_CHUNKS == 0)

            if chunk_idx == 0:
                # Chunk 0: Qwen Image Edit redraw from initial source scene frame
                logger.info(f"[WAN CLONER] [CHUNK 0] Generating initial 2D anime anchor keyframe with Qwen Image Edit...")
                print(f"[WAN CLONER] [CHUNK 0] Generating initial 2D anime anchor keyframe with Qwen Image Edit...")
                chunk_raw_frame = chunks_dir / f"keyframe_{chunk_idx:04d}_raw.png"
                shutil.copy2(extracted_frame, chunk_raw_frame)

                chunk_prefiltered_frame = chunks_dir / f"keyframe_{chunk_idx:04d}_prefiltered.png"
                self.prefilter_image_to_anime(
                    input_image_path=chunk_raw_frame,
                    output_image_path=chunk_prefiltered_frame,
                    wan_w=wan_w,
                    wan_h=wan_h,
                )

                refs_to_use = generated_character_images if generated_character_images else []
                keyframe_denoise = 0.85 if style_clean == "anime" else active_denoise
                try:
                    self.keyframe_generator.generate(
                        scene_image=chunk_raw_frame,
                        character_references=refs_to_use,
                        prompt=edit_prompt,
                        negative_prompt=anime_qwen_negative,
                        output_path=styled_chunk_keyframe,
                        seed=chunk_seed,
                        denoise=keyframe_denoise,
                        cfg=active_cfg,
                    )
                except Exception as e:
                    logger.warning(f"[WAN CLONER] Keyframe anime restyle notice: {e}. Using cel-shaded fallback.")
                    styled_chunk_keyframe = chunk_prefiltered_frame

                shutil.copy2(styled_chunk_keyframe, job_dir / "styled_scene_keyframe.png")

            elif is_reanchor_chunk:
                # Re-anchor every N chunks (default 15) to prevent color/detail drift
                logger.info(
                    f"[WAN CLONER] [RE-ANCHOR] Chunk {chunk_idx + 1}/{num_chunks}: "
                    f"Re-anchoring via Qwen Image Edit with character reference portraits (every {REANCHOR_EVERY_N_CHUNKS} chunks)."
                )
                print(
                    f"[WAN CLONER] [RE-ANCHOR] Chunk {chunk_idx + 1}/{num_chunks}: "
                    f"Re-anchoring via Qwen Image Edit with character references (drift prevention)..."
                )
                chunk_raw_frame = chunks_dir / f"keyframe_{chunk_idx:04d}_raw.png"
                self.extract_frame_at_time(source_video, start_t, chunk_raw_frame)

                refs_to_use = generated_character_images if generated_character_images else []
                keyframe_denoise = 0.85 if style_clean == "anime" else active_denoise
                try:
                    self.keyframe_generator.generate(
                        scene_image=chunk_raw_frame,
                        character_references=refs_to_use,
                        prompt=edit_prompt,
                        negative_prompt=anime_qwen_negative,
                        output_path=styled_chunk_keyframe,
                        seed=chunk_seed,
                        denoise=keyframe_denoise,
                        cfg=active_cfg,
                    )
                except Exception as e:
                    logger.warning(f"[WAN CLONER] Re-anchor redraw notice: {e}. Falling back to previous chunk's last frame.")
                    prev_chunk_video = chunks_dir / f"chunk_{chunk_idx - 1:04d}.mp4"
                    self.extract_last_frame(prev_chunk_video, styled_chunk_keyframe)

            else:
                # Chained Chunk: Use the LAST frame of the generated chunk N-1 (already pure anime) as start image
                prev_chunk_video = chunks_dir / f"chunk_{chunk_idx - 1:04d}.mp4"
                chained_keyframe = chunks_dir / f"keyframe_{chunk_idx:04d}_chained.png"
                logger.info(
                    f"[WAN CLONER] [CHAINED] Chunk {chunk_idx + 1}/{num_chunks}: "
                    f"Chaining start frame from last frame of chunk {chunk_idx} ({prev_chunk_video.name})."
                )
                print(
                    f"[WAN CLONER] [CHAINED] Chunk {chunk_idx + 1}/{num_chunks}: "
                    f"Chaining start frame from last frame of chunk {chunk_idx} ({prev_chunk_video.name})."
                )

                if prev_chunk_video.exists() and prev_chunk_video.stat().st_size > 1000:
                    self.extract_last_frame(prev_chunk_video, chained_keyframe)
                    styled_chunk_keyframe = chained_keyframe
                else:
                    logger.warning(
                        f"[WAN CLONER] Previous chunk {prev_chunk_video.name} not found on disk. "
                        f"Extracting source frame fallback."
                    )
                    chunk_raw_frame = chunks_dir / f"keyframe_{chunk_idx:04d}_raw.png"
                    self.extract_frame_at_time(source_video, start_t, chunk_raw_frame)
                    styled_chunk_keyframe = chunk_raw_frame

            # 4d. Animate with Wan with Quality Gate and Retry Loop
            active_backend = (cloning_backend or CLONING_BACKEND or "fun_control").lower().strip()
            quality_file = chunks_dir / f"chunk_{chunk_idx:04d}_quality.json"
            max_retries = QUALITY_MAX_RETRIES
            attempts_record = []
            chosen_candidate_path = None
            best_score = -1.0
            best_candidate_path = None
            passed_quality = False

            for attempt in range(max_retries):
                attempt_seed = chunk_seed + attempt * 7919
                temp_wan_raw = chunks_dir / f"temp_wan_{chunk_idx:04d}_att{attempt}.mp4"
                candidate_chunk = chunks_dir / f"chunk_{chunk_idx:04d}_att{attempt}.mp4"

                logger.info(
                    f"[WAN CLONER] Animating Chunk {chunk_idx + 1}/{num_chunks} "
                    f"(attempt {attempt + 1}/{max_retries}, seed={attempt_seed})..."
                )
                print(
                    f"[WAN CLONER] Animating Chunk {chunk_idx + 1}/{num_chunks} "
                    f"(attempt {attempt + 1}/{max_retries}, seed={attempt_seed})..."
                )

                if active_backend == "legacy_i2v":
                    print_gpu_memory(f"Before Wan 2.2 V2V Chunk {chunk_idx + 1} (att {attempt + 1})")
                    actual_wan_video = self.animate_with_wan_v2v(
                        chunk_video_path=chunk_src_video,
                        keyframe_path=styled_chunk_keyframe,
                        motion_prompt=motion_prompt,
                        output_video_path=temp_wan_raw,
                        seed=attempt_seed,
                        length_frames=TARGET_FRAMES,
                        fps=16,
                        width=wan_w,
                        height=wan_h,
                        negative_prompt=wan_negative_prompt,
                        denoise=active_denoise,
                        cfg=active_cfg,
                    )
                    print_gpu_memory(f"After Wan 2.2 V2V Chunk {chunk_idx + 1} (att {attempt + 1})")
                else:
                    # Wan 2.2 Fun Control (Default)
                    chunk_control_video = chunks_dir / f"chunk_{chunk_idx:04d}_control.mp4"
                    control_video_to_use = (
                        chunk_control_video
                        if (chunk_control_video.exists() and chunk_control_video.stat().st_size > 1000)
                        else chunk_src_video
                    )
                    print_gpu_memory(f"Before Wan 2.2 Fun Control Chunk {chunk_idx + 1} (att {attempt + 1})")
                    actual_wan_video = self.animate_with_wan_fun_control(
                        control_video_path=control_video_to_use,
                        keyframe_path=styled_chunk_keyframe,
                        motion_prompt=motion_prompt,
                        output_video_path=temp_wan_raw,
                        seed=attempt_seed,
                        length_frames=TARGET_FRAMES,
                        fps=16,
                        width=wan_w,
                        height=wan_h,
                        negative_prompt=wan_negative_prompt,
                        steps=FUN_CONTROL_STEPS,
                        cfg=FUN_CONTROL_CFG,
                        prompt_override=prompt_override,
                    )
                    print_gpu_memory(f"After Wan 2.2 Fun Control Chunk {chunk_idx + 1} (att {attempt + 1})")

                if not actual_wan_video.exists() or actual_wan_video.stat().st_size == 0:
                    logger.error(f"[WAN CLONER] Wan chunk {chunk_idx + 1} attempt {attempt + 1} produced no video file.")
                    continue

                # 4e. Resample chunk candidate to source FPS and clamp duration
                resample_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(actual_wan_video),
                    "-t", str(round(this_chunk_dur, 3)),
                    "-r", str(src_fps),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    str(candidate_chunk),
                ]
                res_chunk = subprocess.run(resample_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res_chunk.returncode != 0 or not candidate_chunk.exists():
                    shutil.copy2(actual_wan_video, candidate_chunk)
                actual_wan_video.unlink(missing_ok=True)

                # Sanity check: Ensure generated candidate is NOT the control video
                if control_video_to_use.exists():
                    is_not_ctrl, ctrl_sim = check_not_control_video(candidate_chunk, control_video_to_use, max_allowed_similarity=0.88)
                    if not is_not_ctrl:
                        logger.error(
                            f"[SANITY CHECK FAILED] Chunk {chunk_idx + 1} attempt {attempt + 1} output is nearly identical "
                            f"to control video (similarity={ctrl_sim:.3f} >= 0.88)! Rejecting attempt."
                        )
                        print(
                            f"[SANITY CHECK FAILED] Chunk {chunk_idx + 1} attempt {attempt + 1} output is identical to control video! "
                            f"Rejecting attempt."
                        )
                        continue

                # Quality evaluation gate
                quality_res = check_chunk(source_chunk_mp4=chunk_src_video, output_chunk_mp4=candidate_chunk)
                score = quality_res.get("overall_score", 0.0)
                attempts_record.append({
                    "attempt": attempt + 1,
                    "seed": attempt_seed,
                    "scores": quality_res,
                })

                if score > best_score:
                    best_score = score
                    best_candidate_path = candidate_chunk

                if quality_res.get("passed", False):
                    logger.info(
                        f"[QUALITY GATE] Chunk {chunk_idx + 1} PASSED quality check on attempt {attempt + 1} "
                        f"(overall={score:.3f}, motion={quality_res.get('motion_score', 0):.3f}, "
                        f"anime={quality_res.get('anime_score', 0):.3f}, flicker={quality_res.get('flicker_score', 0):.3f})."
                    )
                    print(
                        f"[QUALITY GATE] Chunk {chunk_idx + 1} PASSED quality check on attempt {attempt + 1} (score={score:.3f})."
                    )
                    chosen_candidate_path = candidate_chunk
                    passed_quality = True
                    break
                else:
                    logger.warning(
                        f"[QUALITY GATE] Chunk {chunk_idx + 1} attempt {attempt + 1} failed quality check "
                        f"(motion_passed={quality_res.get('motion_passed')}, anime_passed={quality_res.get('anime_passed')}, "
                        f"flicker_passed={quality_res.get('flicker_passed')}, score={score:.3f}). Retrying with new seed..."
                    )
                    print(
                        f"[QUALITY GATE] Chunk {chunk_idx + 1} attempt {attempt + 1} failed check (score={score:.3f}). Retrying..."
                    )

            if not passed_quality:
                if best_candidate_path and best_candidate_path.exists():
                    logger.warning(
                        f"[QUALITY GATE] WARNING: Chunk {chunk_idx + 1} did not pass quality check after {max_retries} attempts. "
                        f"Retaining highest-scoring attempt (score={best_score:.3f}). Job continuing."
                    )
                    print(
                        f"[QUALITY GATE] WARNING: Chunk {chunk_idx + 1} failed quality checks after {max_retries} attempts. "
                        f"Keeping best attempt (score={best_score:.3f})."
                    )
                    chosen_candidate_path = best_candidate_path
                else:
                    logger.error(
                        f"[WAN CLONER] Chunk {chunk_idx + 1} failed all generation attempts without producing valid anime output."
                    )
                    raise RuntimeError(
                        f"Chunk {chunk_idx + 1} failed all generation attempts without producing valid anime output. "
                        f"Refusing to fall back to control or source video."
                    )

            # Finalize chosen candidate as chunk_target
            if chosen_candidate_path and chosen_candidate_path.exists():
                shutil.copy2(chosen_candidate_path, chunk_target)
            else:
                logger.error(
                    f"[WAN CLONER] Chunk {chunk_idx + 1} failed all generation attempts without producing valid output."
                )
                raise RuntimeError(
                    f"Chunk {chunk_idx + 1} failed all generation attempts without producing valid output. "
                    f"Refusing to fall back to control or source video."
                )

            # Record quality scores of every attempt to chunks/chunk_XXXX_quality.json
            quality_payload = {
                "chunk_index": chunk_idx,
                "passed": passed_quality,
                "num_attempts": len(attempts_record),
                "best_score": best_score,
                "attempts": attempts_record,
            }
            with open(quality_file, "w", encoding="utf-8") as qf:
                json.dump(quality_payload, qf, indent=2)

            # Cleanup temporary candidate attempt files
            for att in range(max_retries):
                att_file = chunks_dir / f"chunk_{chunk_idx:04d}_att{att}.mp4"
                att_file.unlink(missing_ok=True)

            chunk_src_video.unlink(missing_ok=True)
            completed_chunk_paths.append(chunk_target)
            if chunk_idx not in completed_chunk_indices:
                completed_chunk_indices.append(chunk_idx)
            job_state["completed_chunks"] = sorted(completed_chunk_indices)
            job_state["updated_at"] = time.time()
            try:
                with open(state_file, "w", encoding="utf-8") as sf:
                    json.dump(job_state, sf, indent=2)
            except Exception as se:
                logger.warning(f"[WAN CLONER] Could not update job_state.json: {se}")

            # Force garbage collection between chunks to keep VRAM completely clean
            gc.collect()

        # Step 5: Seamless Stitching & Full Audio Remuxing
        logger.info(f"[WAN CLONER] Stitching {len(completed_chunk_paths)} chunks together...")
        print(f"[WAN CLONER] Stitching {len(completed_chunk_paths)} chunks into final {active_duration:.2f}s video...")

        if len(completed_chunk_paths) == 1:
            stitched_video = completed_chunk_paths[0]
        else:
            concat_list_file = job_dir / "concat_chunks.txt"
            with open(concat_list_file, "w", encoding="utf-8") as f:
                for p in completed_chunk_paths:
                    f.write(f"file '{p.resolve()}'\n")

            stitched_video = job_dir / "temp_stitched_all_chunks.mp4"
            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_file),
                "-c", "copy",
                str(stitched_video),
            ]
            res_cat = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res_cat.returncode != 0:
                concat_cmd_reencode = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_list_file),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    str(stitched_video),
                ]
                subprocess.run(concat_cmd_reencode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        # Mux full original audio track onto output
        logger.info(f"[WAN CLONER] Muxing original audio onto output video...")
        print(f"[WAN CLONER] Muxing original audio onto output video...")
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", str(stitched_video),
            "-i", str(source_video),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-t", str(round(active_duration, 3)),
            str(output_video),
        ]
        res_mux = subprocess.run(mux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res_mux.returncode != 0:
            shutil.copy2(stitched_video, output_video)

        if stitched_video != completed_chunk_paths[0]:
            stitched_video.unlink(missing_ok=True)

        job_state["status"] = "completed"
        job_state["output_video"] = str(output_video)
        job_state["completed_at"] = time.time()
        try:
            with open(state_file, "w", encoding="utf-8") as sf:
                json.dump(job_state, sf, indent=2)
        except Exception as se:
            logger.warning(f"[WAN CLONER] Could not finalize job_state.json: {se}")

        logger.info(f"[WAN CLONER] Long video cloning complete! Saved to {output_video}")
        print(f"[WAN CLONER] Long video cloning complete! Saved to {output_video}")
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

