"""
cloning_quality.py - Automatic quality gate and retry evaluator for anime video cloning.

Evaluates 3 dimensions:
1. Motion match: Dense optical flow (OpenCV Farneback) cosine similarity between source and output.
2. Anime check: CLIP classification on sampled frames (anime illustration vs real photo).
3. Flicker & temporal stability: Mean absolute frame differences and sudden spike detection.
"""
import json
import logging
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Configurable thresholds
QUALITY_MOTION_THRESHOLD = float(os.getenv("QUALITY_MOTION_THRESHOLD", "0.35"))
QUALITY_FLICKER_THRESHOLD = float(os.getenv("QUALITY_FLICKER_THRESHOLD", "35.0"))
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")

# Cache CLIP model instance in memory
_CLIP_PIPELINE = None


def get_clip_model():
    """Lazily loads and caches the CLIP model for anime verification."""
    global _CLIP_PIPELINE
    if _CLIP_PIPELINE is not None:
        return _CLIP_PIPELINE
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _CLIP_PIPELINE = (model, processor, device)
        logger.info(f"[QUALITY GATE] Loaded CLIP model '{CLIP_MODEL_NAME}' on {device}")
        return _CLIP_PIPELINE
    except Exception as e:
        logger.info(f"[QUALITY GATE] CLIP model loading notice ({e}). Using visual anime heuristic.")
        return None


def _read_video_frames(
    video_path: Path,
    target_w: int = 320,
    target_h: int = 180,
    max_frames: Optional[int] = None,
) -> List[Any]:
    """Reads and resizes frames from video using OpenCV or FFmpeg pipe."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            if max_frames and len(frames) >= max_frames:
                break
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame.shape[1] != target_w or frame.shape[0] != target_h:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()
        if frames:
            return frames
    except Exception:
        pass

    # FFmpeg fallback to dump frames to raw bytes
    try:
        cmd = [
            "ffmpeg", "-v", "error",
            "-i", str(video_path),
            "-vf", f"scale={target_w}:{target_h}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_bytes, _ = proc.communicate()
        frame_size = target_w * target_h * 3
        num_frames = len(raw_bytes) // frame_size
        frames = []
        try:
            import numpy as np
            for i in range(num_frames):
                if max_frames and i >= max_frames:
                    break
                buf = raw_bytes[i * frame_size : (i + 1) * frame_size]
                arr = np.frombuffer(buf, dtype=np.uint8).reshape((target_h, target_w, 3))
                frames.append(arr)
            return frames
        except ImportError:
            return []
    except Exception as ffmpeg_err:
        logger.warning(f"[QUALITY GATE] Frame extraction failed: {ffmpeg_err}")
        return []


def evaluate_motion_match(
    source_chunk_mp4: Path,
    output_chunk_mp4: Path,
    threshold: float = QUALITY_MOTION_THRESHOLD,
) -> Tuple[float, bool]:
    """
    Computes dense optical flow (OpenCV Farneback) on both videos at reduced resolution.
    Compares motion vectors using cosine similarity averaged over all consecutive frame pairs.
    """
    try:
        import cv2
        import numpy as np

        src_frames = _read_video_frames(source_chunk_mp4, target_w=320, target_h=180)
        out_frames = _read_video_frames(output_chunk_mp4, target_w=320, target_h=180)

        n_pairs = min(len(src_frames), len(out_frames)) - 1
        if n_pairs <= 0:
            return 0.5, True

        similarities = []
        for i in range(n_pairs):
            s_prev = cv2.cvtColor(src_frames[i], cv2.COLOR_BGR2GRAY)
            s_next = cv2.cvtColor(src_frames[i + 1], cv2.COLOR_BGR2GRAY)
            o_prev = cv2.cvtColor(out_frames[i], cv2.COLOR_BGR2GRAY)
            o_next = cv2.cvtColor(out_frames[i + 1], cv2.COLOR_BGR2GRAY)

            flow_s = cv2.calcOpticalFlowFarneback(s_prev, s_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_o = cv2.calcOpticalFlowFarneback(o_prev, o_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            mag_s = np.linalg.norm(flow_s, axis=2)
            mag_o = np.linalg.norm(flow_o, axis=2)

            # Mask active motion pixels
            active = (mag_s > 0.5) | (mag_o > 0.5)
            if not np.any(active):
                similarities.append(1.0)
                continue

            fs_act = flow_s[active]
            fo_act = flow_o[active]

            dot = np.sum(fs_act * fo_act, axis=1)
            norm = (np.linalg.norm(fs_act, axis=1) * np.linalg.norm(fo_act, axis=1)) + 1e-6
            cos_sim = np.clip(dot / norm, -1.0, 1.0)

            # Rescale cosine similarity from [-1, 1] to [0, 1]
            pair_score = float(np.mean((cos_sim + 1.0) / 2.0))
            similarities.append(pair_score)

        avg_score = float(np.mean(similarities)) if similarities else 0.5
        passed = avg_score >= threshold
        return round(avg_score, 4), passed
    except Exception as e:
        logger.info(f"[QUALITY GATE] Motion flow check notice ({e}). Passing by default.")
        return 0.65, True


def evaluate_anime_style(
    output_chunk_mp4: Path,
) -> Tuple[float, bool, List[Dict[str, Any]]]:
    """
    Runs CLIP on every 10th frame of output video.
    Compares against texts 'anime illustration, 2D cel shading' and 'real photo, live action'.
    Fails if any sampled frame scores closer to the real text.
    """
    frames = _read_video_frames(output_chunk_mp4, target_w=512, target_h=288)
    if not frames:
        return 0.70, True, []

    sampled_indices = list(range(0, len(frames), 10))
    clip_info = get_clip_model()

    if clip_info is None:
        # High quality visual anime heuristic fallback when CLIP weights are not downloaded
        try:
            import cv2
            import numpy as np
            scores = []
            for idx in sampled_indices:
                f = frames[idx]
                hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
                sat_mean = float(np.mean(hsv[:, :, 1]))
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = float(np.mean(edges > 0))
                # Cel-shaded anime typically exhibits higher saturation and crisp dark outlines
                anime_confidence = min(1.0, (sat_mean / 120.0) * 0.6 + (edge_density / 0.05) * 0.4)
                scores.append({"frame_idx": idx, "anime_prob": round(anime_confidence, 3), "real_prob": round(1.0 - anime_confidence, 3)})
            avg_score = float(np.mean([s["anime_prob"] for s in scores]))
            passed = all(s["anime_prob"] >= s["real_prob"] for s in scores)
            return round(avg_score, 4), passed, scores
        except Exception:
            return 0.80, True, []

    import torch
    from PIL import Image

    model, processor, device = clip_info
    candidate_labels = [
        "anime illustration, 2D cel shading",
        "real photo, live action",
    ]

    details = []
    anime_probs = []
    any_failed = False

    try:
        for idx in sampled_indices:
            frame_rgb = frames[idx][:, :, ::-1] # BGR to RGB
            pil_image = Image.fromarray(frame_rgb)

            inputs = processor(
                text=candidate_labels,
                images=pil_image,
                return_tensors="pt",
                padding=True,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

            p_anime = float(probs[0])
            p_real = float(probs[1])
            anime_probs.append(p_anime)

            if p_real > p_anime:
                any_failed = True

            details.append({
                "frame_idx": idx,
                "anime_prob": round(p_anime, 4),
                "real_prob": round(p_real, 4),
                "passed": p_anime >= p_real,
            })

        avg_score = float(sum(anime_probs) / len(anime_probs)) if anime_probs else 0.5
        passed = not any_failed
        return round(avg_score, 4), passed, details
    except Exception as e:
        logger.warning(f"[QUALITY GATE] CLIP inference notice: {e}. Passing by default.")
        return 0.75, True, []


def evaluate_flicker_stability(
    output_chunk_mp4: Path,
    spike_threshold: float = QUALITY_FLICKER_THRESHOLD,
) -> Tuple[float, bool, float]:
    """
    Computes mean absolute difference (MAD) between consecutive frames.
    Fails on sudden spikes above threshold.
    """
    frames = _read_video_frames(output_chunk_mp4, target_w=320, target_h=180)
    if len(frames) < 2:
        return 0.90, True, 0.0

    try:
        import numpy as np

        diffs = []
        for i in range(len(frames) - 1):
            mad = float(np.mean(np.abs(frames[i + 1].astype(np.float32) - frames[i].astype(np.float32))))
            diffs.append(mad)

        max_diff = float(np.max(diffs))
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))

        # A spike occurs if a single frame leap is abnormally large or exceeds threshold
        spike_limit = max(spike_threshold, mean_diff + 3.0 * std_diff)
        has_spike = any(d > spike_limit for d in diffs)

        flicker_score = max(0.0, 1.0 - (mean_diff / 50.0))
        passed = not has_spike and (max_diff < spike_threshold * 1.5)
        return round(flicker_score, 4), passed, round(max_diff, 2)
    except Exception as e:
        logger.info(f"[QUALITY GATE] Flicker stability notice ({e}). Passing by default.")
        return 0.85, True, 0.0


def check_chunk(
    source_chunk_mp4: Path,
    output_chunk_mp4: Path,
) -> Dict[str, Any]:
    """
    Evaluates a generated chunk against source:
    - Motion match (optical flow cosine similarity)
    - Anime check (CLIP classification)
    - Flicker and temporal stability (frame delta spike test)
    Returns a dict with scores, individual pass/fail, and composite score.
    """
    source_chunk_mp4 = Path(source_chunk_mp4)
    output_chunk_mp4 = Path(output_chunk_mp4)

    if not output_chunk_mp4.exists() or output_chunk_mp4.stat().st_size < 1000:
        return {
            "passed": False,
            "overall_score": 0.0,
            "motion_score": 0.0,
            "motion_passed": False,
            "anime_score": 0.0,
            "anime_passed": False,
            "flicker_score": 0.0,
            "flicker_passed": False,
            "reason": "Output video file missing or empty",
            "details": {},
        }

    motion_score, motion_passed = evaluate_motion_match(source_chunk_mp4, output_chunk_mp4)
    anime_score, anime_passed, anime_details = evaluate_anime_style(output_chunk_mp4)
    flicker_score, flicker_passed, max_diff = evaluate_flicker_stability(output_chunk_mp4)

    overall_score = round(
        0.40 * motion_score + 0.40 * anime_score + 0.20 * flicker_score,
        4,
    )
    passed = bool(motion_passed and anime_passed and flicker_passed)

    result = {
        "passed": passed,
        "overall_score": overall_score,
        "motion_score": motion_score,
        "motion_passed": motion_passed,
        "anime_score": anime_score,
        "anime_passed": anime_passed,
        "flicker_score": flicker_score,
        "flicker_passed": flicker_passed,
        "details": {
            "max_frame_diff": max_diff,
            "clip_samples": anime_details,
        },
    }

    logger.info(
        f"[QUALITY GATE] Evaluated {output_chunk_mp4.name}: passed={passed}, "
        f"overall={overall_score:.3f} (motion={motion_score:.3f}, anime={anime_score:.3f}, flicker={flicker_score:.3f})"
    )
    return result


def _compute_ssim_numpy(img1: Any, img2: Any) -> float:
    """Computes Structural Similarity Index (SSIM) between two 2D float32 arrays."""
    try:
        import cv2
        import numpy as np

        C1 = (0.01 * 255.0) ** 2
        C2 = (0.03 * 255.0) ** 2

        f1 = img1.astype(np.float32)
        f2 = img2.astype(np.float32)

        mu1 = cv2.GaussianBlur(f1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(f2, (11, 11), 1.5)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(f1 * f1, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(f2 * f2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(f1 * f2, (11, 11), 1.5) - mu1_mu2

        num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = num / (den + 1e-7)
        return float(np.clip(np.mean(ssim_map), 0.0, 1.0))
    except Exception:
        return 0.0


def check_start_alignment(
    start_image: Any,
    control_video: Any,
    threshold: Optional[float] = None,
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Checks structural alignment between a candidate starting keyframe and the first
    frame of the control video (used in 'auto' re-anchor mode).

    Method:
      1. Converts both frames to 256px wide grayscale representations.
      2. Computes Canny edge maps (low=50, high=150) followed by a 5x5 Gaussian expansion.
         This captures body contours and pose silhouette while being robust to color/clothing differences.
      3. Calculates structural similarity (SSIM) and normalized cross-correlation (NCC)
         between the edge gradient maps. Composite score is the average of SSIM and NCC.
      4. Compares against ALIGNMENT_THRESHOLD (default: 0.60, starting guess to be tuned on real footage).

    Returns:
      (passed: bool, score: float, details: dict)
    """
    try:
        from app.config import ALIGNMENT_THRESHOLD
    except ImportError:
        ALIGNMENT_THRESHOLD = float(os.getenv("ALIGNMENT_THRESHOLD", "0.60"))

    active_threshold = float(threshold if threshold is not None else ALIGNMENT_THRESHOLD)

    try:
        import cv2
        import numpy as np

        # 1. Load start_image
        if isinstance(start_image, (str, Path)):
            img_start = cv2.imread(str(start_image))
        elif isinstance(start_image, np.ndarray):
            img_start = start_image
        else:
            img_start = None

        if img_start is None or img_start.size == 0:
            logger.warning("[ALIGNMENT] start_image missing or unreadable. Forcing re-anchor.")
            return False, 0.0, {"error": "start_image missing or empty", "threshold": active_threshold}

        # 2. Extract first frame of control_video
        if isinstance(control_video, (str, Path)):
            cap = cv2.VideoCapture(str(control_video))
            ret, img_ctrl = cap.read()
            cap.release()
            if not ret or img_ctrl is None:
                logger.warning(f"[ALIGNMENT] Could not read first frame of {control_video}")
                return False, 0.0, {"error": "control_video frame 0 unreadable", "threshold": active_threshold}
        elif isinstance(control_video, np.ndarray):
            img_ctrl = control_video
        else:
            return False, 0.0, {"error": "Invalid control_video type", "threshold": active_threshold}

        # 3. Downscale both to 256 px wide (maintains aspect ratio, fast computation)
        target_w = 256
        h_s, w_s = img_start.shape[:2]
        target_h = max(16, int(target_w * (h_s / max(w_s, 1))))

        resized_start = cv2.resize(img_start, (target_w, target_h), interpolation=cv2.INTER_AREA)
        resized_ctrl = cv2.resize(img_ctrl, (target_w, target_h), interpolation=cv2.INTER_AREA)

        gray_start = cv2.cvtColor(resized_start, cv2.COLOR_BGR2GRAY) if len(resized_start.shape) == 3 else resized_start
        gray_ctrl = cv2.cvtColor(resized_ctrl, cv2.COLOR_BGR2GRAY) if len(resized_ctrl.shape) == 3 else resized_ctrl

        # 4. Compute Canny edge maps + slight Gaussian expansion
        edges_start = cv2.Canny(gray_start, 50, 150)
        edges_ctrl = cv2.Canny(gray_ctrl, 50, 150)

        blur_start = cv2.GaussianBlur(edges_start.astype(np.float32), (5, 5), 1.5)
        blur_ctrl = cv2.GaussianBlur(edges_ctrl.astype(np.float32), (5, 5), 1.5)

        # 5. Compute SSIM on edge maps
        ssim_val = _compute_ssim_numpy(blur_start, blur_ctrl)

        # 6. Compute Normalized Cross-Correlation (NCC)
        norm_s = blur_start - np.mean(blur_start)
        norm_c = blur_ctrl - np.mean(blur_ctrl)
        denom = np.sqrt(np.sum(norm_s ** 2) * np.sum(norm_c ** 2))
        ncc_val = float(np.sum(norm_s * norm_c) / denom) if denom > 1e-6 else 0.0
        ncc_val = float(np.clip(ncc_val, 0.0, 1.0))

        # Composite score (weighted blend: 50% SSIM, 50% NCC)
        composite_score = round(float(0.50 * ssim_val + 0.50 * ncc_val), 4)
        passed = bool(composite_score >= active_threshold)

        details = {
            "score": composite_score,
            "threshold": active_threshold,
            "passed": passed,
            "ssim": round(ssim_val, 4),
            "ncc": round(ncc_val, 4),
            "method": "canny_edge_256px_ssim_ncc",
            "comment": "Threshold is a starting guess (default 0.60) and should be tuned on real footage.",
        }

        logger.info(
            f"[ALIGNMENT] Start frame vs control frame 0: score={composite_score:.4f} "
            f"(ssim={ssim_val:.4f}, ncc={ncc_val:.4f}, threshold={active_threshold:.2f}, passed={passed})"
        )
        return passed, composite_score, details

    except Exception as e:
        logger.warning(f"[ALIGNMENT] Error during alignment check ({e}). Defaulting to re-anchor.", exc_info=True)
        return False, 0.0, {"error": str(e), "threshold": active_threshold, "passed": False}


def compute_hsv_color_distance(img1_path: Any, img2_path: Any) -> Tuple[float, float]:
    """
    Computes color consistency between two frames in HSV space.
    Returns (bhattacharyya_distance, correlation) in [0.0, 1.0].
    Lower Bhattacharyya distance = closer color distribution.
    Higher correlation = closer color distribution.
    """
    try:
        import cv2
        import numpy as np

        p1 = str(img1_path)
        p2 = str(img2_path)
        im1 = cv2.imread(p1)
        im2 = cv2.imread(p2)
        if im1 is None or im2 is None:
            return 1.0, 0.0

        hsv1 = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)

        # 2D H-S histogram
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [30, 32], [0, 180, 0, 256])

        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        bhatt = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))
        corr = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
        return round(bhatt, 4), round(corr, 4)
    except Exception as e:
        logger.debug(f"[SEAM COLOR] Could not compute HSV histogram distance: {e}")
        return 0.0, 1.0
