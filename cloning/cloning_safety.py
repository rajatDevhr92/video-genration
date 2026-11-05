"""
cloning_safety.py - content and age check that runs before any cloning job.

The check fails closed: if a detector cannot load or run, the job is rejected.
Protects against underage media while supporting adult productions.
"""
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Age limit sits safely at 18 or above (default 21 to avoid borderline false-negatives)
MIN_APPARENT_AGE = float(os.getenv("CLONING_MIN_APPARENT_AGE", "21"))
SAMPLE_FPS = float(os.getenv("CLONING_SAMPLE_FPS", "1"))


def sample_frames(video_path, output_dir) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={SAMPLE_FPS}",
            str(output_dir / "frame_%06d.png"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg could not read the video:\n" + result.stderr
        )

    return sorted(output_dir.glob("frame_*.png"))


@lru_cache(maxsize=1)
def _face_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def default_face_ages(image_path) -> List[float]:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    return [float(face.age) for face in _face_app().get(image)]


def _review_image(image_path, label, face_ages) -> Dict:
    reasons = []
    ages = face_ages(image_path)
    for age in ages:
        if age < MIN_APPARENT_AGE:
            reasons.append(f"{label}: person with apparent age {age:.0f} (below limit {MIN_APPARENT_AGE:.0f})")

    return {"reasons": reasons, "faces": len(ages)}


def review_media(
    source_video,
    face_ages: Optional[Callable] = None,
    frame_sampler: Callable = sample_frames,
) -> Dict:
    """
    Check sampled source frames for adult-age compliance.
    Returns {"passed": bool, "reasons": [str], "frames_checked": int}.
    """
    face_ages = face_ages or default_face_ages

    reasons: List[str] = []
    frames_checked = 0

    try:
        with tempfile.TemporaryDirectory(prefix="cloning_frames_") as temp_dir:
            frames = frame_sampler(source_video, temp_dir)

            if not frames:
                reasons.append("source video: no frames could be read")

            source_faces = 0
            for frame in frames:
                result = _review_image(
                    frame,
                    f"source video {Path(frame).name}",
                    face_ages,
                )
                reasons.extend(result["reasons"])
                source_faces += result["faces"]
                frames_checked += 1

            if frames and source_faces == 0:
                reasons.append("source video: no face found in sampled frames")

    except Exception as error:
        return {
            "passed": False,
            "reasons": [f"safety check could not run: {error}"],
            "frames_checked": frames_checked,
        }

    return {
        "passed": not reasons,
        "reasons": reasons,
        "frames_checked": frames_checked,
    }
