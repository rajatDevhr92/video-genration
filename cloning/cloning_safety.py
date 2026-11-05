"""
cloning_safety.py - content and age check that runs before any cloning job.

The check fails closed: if a detector cannot load or run, the job is rejected.
Automated checks can miss things, so a pass is not proof that the media is safe.
"""
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

NSFW_MODEL = os.getenv("CLONING_NSFW_MODEL", "Falconsai/nsfw_image_detection")
NSFW_THRESHOLD = float(os.getenv("CLONING_NSFW_THRESHOLD", "0.30"))

# Age estimates can be off by several years, so the limit sits well above 18.
MIN_APPARENT_AGE = float(os.getenv("CLONING_MIN_APPARENT_AGE", "25"))

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
def _nsfw_pipeline():
    from transformers import pipeline

    return pipeline("image-classification", model=NSFW_MODEL)


def default_nsfw_score(image_path) -> float:
    for item in _nsfw_pipeline()(str(image_path)):
        if item["label"].lower() == "nsfw":
            return float(item["score"])

    raise RuntimeError(f"{NSFW_MODEL} returned no 'nsfw' label.")


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


def _review_image(image_path, label, nsfw_score, face_ages) -> Dict:
    reasons = []

    score = nsfw_score(image_path)
    if score >= NSFW_THRESHOLD:
        reasons.append(f"{label}: sexual content score {score:.2f}")

    ages = face_ages(image_path)
    for age in ages:
        if age < MIN_APPARENT_AGE:
            reasons.append(f"{label}: person with apparent age {age:.0f}")

    return {"reasons": reasons, "faces": len(ages)}


def review_media(
    source_video,
    character_image,
    nsfw_score: Optional[Callable] = None,
    face_ages: Optional[Callable] = None,
    frame_sampler: Callable = sample_frames,
) -> Dict:
    """
    Check every sampled source frame and the character image.
    Returns {"passed": bool, "reasons": [str], "frames_checked": int}.
    Sampled frames live in a temporary folder and are never kept.
    """
    nsfw_score = nsfw_score or default_nsfw_score
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
                    nsfw_score,
                    face_ages,
                )
                reasons.extend(result["reasons"])
                source_faces += result["faces"]
                frames_checked += 1

            if frames and source_faces == 0:
                reasons.append("source video: no face found, so age cannot be checked")

        character = _review_image(
            character_image,
            "character image",
            nsfw_score,
            face_ages,
        )
        reasons.extend(character["reasons"])

        if character["faces"] == 0:
            reasons.append("character image: no face found, so age cannot be checked")

    except Exception as error:
        reasons.append(f"safety check could not run: {error}")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "frames_checked": frames_checked,
    }
