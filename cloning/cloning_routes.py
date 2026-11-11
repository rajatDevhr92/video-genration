"""
cloning_routes.py - API routes for AI Character Face Replacement & Video Cloning.
"""
import json
import logging
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import BASE_DIR, OUTPUT_DIR
from app.composition_generator import CompositionGenerator

from .cloning_record import CloningRecord
from .cloning_safety import review_media

logger = logging.getLogger(__name__)

CLONING_INPUT_DIR = BASE_DIR / "input" / "cloning"
CLONING_INPUT_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(
    prefix="/api/v1/cloning",
    tags=["cloning"],
)

executor = ThreadPoolExecutor(max_workers=1)


class CharacterTarget(BaseModel):
    gender: str = Field(
        default="female",
        description="'female' or 'male' to match actors in video"
    )
    age: Optional[int] = Field(
        default=25,
        ge=18,
        description="Target age of character (must be >= 18)"
    )
    character_prompt: Optional[str] = Field(
        default=None,
        description="Optional: Custom prompt for generating face. If null, auto-generated based on gender and age."
    )
    style: Optional[str] = Field(
        default="anime",
        description="Visual style: 'anime' (default) or 'realistic'"
    )
    target_id: Optional[str] = Field(
        default=None,
        description="Optional tag for multiple characters"
    )


class CloningJobRequest(BaseModel):
    source_video: str = Field(
        ...,
        description="File name inside input/cloning on the server (e.g. 'input_clip.mp4')."
    )
    characters: Optional[List[CharacterTarget]] = Field(
        default=None,
        description="Optional: List of characters in the video to replace."
    )
    mode: Optional[str] = Field(
        default="wan_v2v",
        description="Cloning mode: 'wan_v2v' (default: full Wan 2.2 diffusion video cloning) or 'face_swap'"
    )


def resolve_input(name: str) -> Path:
    input_dir = CLONING_INPUT_DIR.resolve()
    path = (input_dir / name).resolve()

    if not path.is_relative_to(input_dir):
        raise HTTPException(
            status_code=422,
            detail=f"File must be inside input/cloning: {name}",
        )

    if not path.is_file():
        raise HTTPException(
            status_code=422,
            detail=f"File not found in input/cloning: {name}",
        )

    return path


def write_status(job_dir: Path, **values):
    status_file = job_dir / "cloning_status.json"
    data = {}
    if status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass
    data.update(values)
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def auto_generate_prompts_for_character_angles(gender: str, age: int, base_prompt: Optional[str] = None) -> Dict[str, str]:
    """
    Returns separate prompts for front and 3/4 side angles.
    """
    safe_age = max(int(age), 21)
    if base_prompt and base_prompt.strip():
        core = base_prompt.strip().rstrip(".")
        if "portrait" in core.lower() or "headshot" in core.lower():
            return {
                "front": f"{core}, looking directly at camera, front facing, 8k masterpiece",
                "side": f"{core}, 3/4 angle side view, dynamic lighting, sharp focus, 8k masterpiece"
            }
        return {
            "front": f"cinematic close-up portrait headshot of {core}, looking straight directly at camera, front facing, 8k masterpiece",
            "side": f"cinematic close-up portrait headshot of {core}, 3/4 angle side view, dynamic lighting, sharp focus, 8k masterpiece"
        }

    if gender.lower() == "female":
        return {
            "front": (
                f"cinematic close-up portrait headshot of an attractive {safe_age}-year-old adult woman, "
                f"looking directly at camera, front view, natural skin texture, detailed eyes, 8k masterpiece"
            ),
            "side": (
                f"cinematic close-up portrait headshot of an attractive {safe_age}-year-old adult woman, "
                f"3/4 angle side view, elegant jawline, soft volumetric lighting, 8k masterpiece"
            )
        }
    else:
        return {
            "front": (
                f"cinematic close-up portrait headshot of a handsome {safe_age}-year-old adult man, "
                f"looking directly at camera, front view, clean haircut, sharp focus, 8k masterpiece"
            ),
            "side": (
                f"cinematic close-up portrait headshot of a handsome {safe_age}-year-old adult man, "
                f"3/4 angle side view, defined jawline, dramatic cinematic lighting, 8k masterpiece"
            )
        }


def auto_generate_prompt_for_character(gender: str, age: int, base_prompt: Optional[str] = None) -> str:
    """Convenience single-prompt generator (front view)."""
    return auto_generate_prompts_for_character_angles(gender, age, base_prompt)["front"]


def process_cloning_job(
    job_id: str,
    job_dir: Path,
    source_video: Path,
    characters: Optional[List[CharacterTarget]],
    mode: str = "wan_v2v",
):
    """
    Background worker:
    mode='wan_v2v' (default):
      Auto-detects video resolution & framerate (1080p, 30 FPS).
      Anchors character identity into the real scene and animates in high-res Wan 2.2 diffusion.
    mode='face_swap':
      Preserves 100% of the original video (background, room, bodies, all motions, actions, audio)
      and swaps only the character's face.
    """
    if mode in ("wan_v2v", "wan_animation", "wan"):
        try:
            from .wan_video_cloner import WanVideoCloner
            cloner = WanVideoCloner()

            style = "anime"
            prompts = []
            if characters:
                for c in characters:
                    if getattr(c, "style", None):
                        style = c.style.lower().strip()
                    if c.character_prompt and c.character_prompt.strip():
                        prompts.append(c.character_prompt.strip())
                    else:
                        if style == "anime":
                            prompts.append(f"gorgeous attractive adult anime {c.gender} with expressive detailed anime eyes and stylized anime hair")
                        else:
                            prompts.append(f"attractive {c.age}yo adult {c.gender} with authentic human features and natural skin texture")
            else:
                prompts.append("attractive adult anime couple, vibrant anime aesthetic, masterpiece cinematography")

            write_status(job_dir, job_id=job_id, status=f"generating_{style}_keyframe")

            # Step 2: Animate with Wan 2.2 I2V
            write_status(job_dir, job_id=job_id, status="animating_with_wan")
            output_video_path = job_dir / "cloned_output.mp4"

            char_meta = []
            if characters:
                for c in characters:
                    char_meta.append({
                        "gender": c.gender.lower() if c.gender else "unknown",
                        "age": c.age or 25,
                        "character_prompt": c.character_prompt or "",
                        "style": getattr(c, "style", "realistic") or "realistic",
                    })

            logger.info(f"[WAN CLONING] Running Wan 2.2 Cloner ({style}) on {source_video}")
            cloner.clone_video(
                source_video=source_video,
                character_prompts=prompts,
                output_video=output_video_path,
                job_dir=job_dir,
                seed=int(uuid.uuid4().int % 1000000),
                style=style,
                characters=char_meta,
            )

            # Scan generated character faces in job_dir to include in status
            char_images = sorted(list(job_dir.glob("character_*_solo_face.png")))
            char_image_urls = [f"/output/{job_id}/{p.name}" for p in char_images]
            styled_keyframe = job_dir / "styled_scene_keyframe.png"
            keyframe_url = f"/output/{job_id}/{styled_keyframe.name}" if styled_keyframe.exists() else None

            write_status(
                job_dir,
                job_id=job_id,
                status="completed",
                output_video=str(output_video_path),
                relative_video_url=f"/output/{job_id}/cloned_output.mp4",
                model_used="Wan 2.2 I2V 14B + Lightning LoRA",
                character_images=char_image_urls,
                scene_keyframe_url=keyframe_url,
            )
        except Exception as e:
            logger.exception(f"[WAN CLONING] Critical error during job {job_id}")
            write_status(
                job_dir,
                job_id=job_id,
                status="failed",
                error=str(e),
            )
        return

    # -------------------------------------------------------------------------
    # Default: Face Swap (Preserves 100% of original video, motions, and background)
    # -------------------------------------------------------------------------
    write_status(job_dir, job_id=job_id, status="analyzing_video_actors")
    try:
        from .cloning_generator import FaceSwapGenerator
        generator = FaceSwapGenerator()
    except ImportError as ie:
        write_status(
            job_dir,
            job_id=job_id,
            status="failed",
            error=f"Face swap dependencies missing on server: {ie}. Run: pip install opencv-python-headless insightface onnxruntime-gpu",
        )
        return

    try:
        logger.info(f"[CLONING] Starting actor analysis on {source_video}...")
        detected_clusters = generator.analyze_source_video(source_video, max_sample_frames=30)

        if not detected_clusters:
            write_status(
                job_dir,
                job_id=job_id,
                status="failed",
                error="No clear faces detected in source video to clone. Ensure the character's face appears in at least some frames.",
            )
            return

        logger.info(f"[CLONING] Detected {len(detected_clusters)} actor cluster(s).")
        write_status(job_dir, job_id=job_id, status="generating_ai_faces", detected_actors=len(detected_clusters))

        comp_gen = CompositionGenerator()
        generated_faces_dir = job_dir / "generated_faces"
        generated_faces_dir.mkdir(parents=True, exist_ok=True)

        spec_list = characters if characters else []
        cluster_to_target_face = {}
        cluster_centroids = {}

        for cluster in detected_clusters:
            c_id = cluster["cluster_id"]
            c_gender = cluster["gender"]
            c_age = cluster["age"]
            cluster_centroids[c_id] = cluster["embedding"]

            # Match character spec by gender or order
            matching_spec = None
            for spec in spec_list:
                if spec.gender.lower() == c_gender.lower():
                    matching_spec = spec
                    break
            if not matching_spec and spec_list:
                matching_spec = spec_list[min(c_id, len(spec_list) - 1)]

            custom_p = matching_spec.character_prompt if (matching_spec and matching_spec.character_prompt) else None
            target_age = matching_spec.age if (matching_spec and matching_spec.age) else c_age
            angle_prompts = auto_generate_prompts_for_character_angles(c_gender, target_age, base_prompt=custom_p)

            target_face_angles = {}
            for angle_name, char_prompt in angle_prompts.items():
                char_face_path = generated_faces_dir / f"character_cluster_{c_id}_{c_gender}_{angle_name}.png"
                seed_offset = 0 if angle_name == "front" else 500
                try:
                    logger.info(f"[CLONING] Generating AI face ({angle_name}) for cluster {c_id} ({c_gender})")
                    comp_gen.generate_simple(
                        prompt=char_prompt,
                        output_path=char_face_path,
                        seed=100000 + c_id * 7919 + seed_offset,
                    )
                    target_face_obj = generator.extract_face_from_image(char_face_path)
                    target_face_angles[angle_name] = target_face_obj
                except Exception as e:
                    logger.warning(f"[CLONING] Could not generate face angle {angle_name} for cluster {c_id}: {e}")

            if target_face_angles:
                cluster_to_target_face[c_id] = target_face_angles

        if not cluster_to_target_face:
            write_status(
                job_dir,
                job_id=job_id,
                status="failed",
                error="Failed to generate AI character faces for the detected actors.",
            )
            return

        # Frame-by-frame face replacement on original video
        write_status(job_dir, job_id=job_id, status="swapping_faces")
        output_video_path = job_dir / "cloned_output.mp4"

        cluster_genders = {c["cluster_id"]: c["gender"] for c in detected_clusters}
        generator.swap_video_faces(
            source_video=source_video,
            mapping=cluster_to_target_face,
            cluster_centroids=cluster_centroids,
            output_video=output_video_path,
            cluster_genders=cluster_genders,
        )

        write_status(
            job_dir,
            job_id=job_id,
            status="completed",
            output_video=str(output_video_path),
            relative_video_url=f"/output/{job_id}/cloned_output.mp4",
            model_used="InsightFace + InSwapper (Original Motion, Bodies & Background 100% Preserved)",
            detected_actors=len(detected_clusters),
        )
    except Exception as e:
        logger.exception(f"[CLONING] Critical error during job {job_id}")
        write_status(
            job_dir,
            job_id=job_id,
            status="failed",
            error=str(e),
        )


@router.post("/jobs")
def create_cloning_job(request: CloningJobRequest):
    source_video = resolve_input(request.source_video)

    job_id = f"clone_{uuid.uuid4().hex[:12]}"
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the input source video directly into the job's output directory
    input_video_dest = job_dir / f"input_{source_video.name}"
    try:
        shutil.copy2(source_video, input_video_dest)
        active_source_video = input_video_dest
    except Exception as copy_err:
        logger.warning(f"Could not copy source video into {job_dir}: {copy_err}")
        active_source_video = source_video

    with open(job_dir / "cloning_record.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_video": request.source_video,
                "input_video_copy": str(active_source_video),
                "characters": [c.model_dump() for c in request.characters],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    write_status(
        job_dir,
        job_id=job_id,
        status="queued",
        input_video_url=f"/output/{job_id}/{active_source_video.name}",
    )

    executor.submit(
        process_cloning_job,
        job_id,
        job_dir,
        active_source_video,
        request.characters,
        request.mode or "face_swap",
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "input_video_url": f"/output/{job_id}/{active_source_video.name}",
        "status_url": f"/api/v1/cloning/jobs/{job_id}",
    }


@router.post("/upload")
async def create_cloning_job_upload(
    video: UploadFile = File(..., description="The MP4 video file to upload directly"),
    characters: Optional[str] = Form(
        default=None,
        description='Optional JSON string array of characters: e.g. [{"gender":"female","age":25,"character_prompt":"..."}]'
    ),
    mode: Optional[str] = Form(
        default="wan_v2v",
        description="Cloning mode: 'wan_v2v' (default: full Wan 2.2 diffusion video cloning) or 'face_swap'"
    ),
):
    """
    Direct multipart/form-data upload: upload the video file directly with character configurations.
    """
    if not video.filename:
        raise HTTPException(status_code=422, detail="No video file provided.")

    char_targets = []
    if characters and characters.strip():
        try:
            char_data = json.loads(characters)
            if isinstance(char_data, dict):
                char_data = [char_data]
            char_targets = [CharacterTarget(**c) for c in char_data]
        except Exception as err:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid 'characters' JSON form field: {err}. Expected format: [{{'gender':'female','age':25}}]"
            )

    job_id = f"clone_{uuid.uuid4().hex[:12]}"
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video directly into the job's directory
    safe_name = Path(video.filename).name
    saved_video_path = job_dir / f"input_{safe_name}"

    try:
        with open(saved_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    finally:
        await video.close()

    active_mode = (mode or "face_swap").lower().strip()

    with open(job_dir / "cloning_record.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_video": safe_name,
                "input_video_copy": str(saved_video_path),
                "characters": [c.model_dump() for c in char_targets],
                "mode": active_mode,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    write_status(
        job_dir,
        job_id=job_id,
        status="queued",
        input_video_url=f"/output/{job_id}/{saved_video_path.name}",
        mode=active_mode,
    )

    executor.submit(
        process_cloning_job,
        job_id,
        job_dir,
        saved_video_path,
        char_targets,
        active_mode,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "input_video_url": f"/output/{job_id}/{saved_video_path.name}",
        "status_url": f"/api/v1/cloning/jobs/{job_id}",
    }


@router.get("/jobs/{job_id}")
def cloning_job_status(job_id: str):
    if not job_id.startswith("clone_") or Path(job_id).name != job_id:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    status_path = OUTPUT_DIR / job_id / "cloning_status.json"

    if not status_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    with open(status_path, "r", encoding="utf-8") as f:
        return json.load(f)
