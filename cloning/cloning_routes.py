"""
cloning_routes.py - API routes for AI Character Face Replacement & Video Cloning.
"""
import json
import logging
import shutil
import subprocess
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
    source_video: Optional[str] = Field(
        default=None,
        description="File name inside input/cloning or input on the server (e.g. 'Test5.mp4')."
    )
    video_path: Optional[str] = Field(
        default=None,
        description="Alias for source_video (e.g. 'input/Test5.mp4' or 'Test5.mp4')."
    )
    characters: Optional[List[CharacterTarget]] = Field(
        default=None,
        description="Optional: List of characters in the video to replace."
    )
    mode: Optional[str] = Field(
        default="wan_v2v",
        description="Cloning mode: 'wan_v2v' (default: full Wan 2.2 diffusion video cloning) or 'face_swap'"
    )
    max_duration: Optional[float] = Field(
        default=None,
        description="Optional: Max seconds of video to process (defaults to full video, e.g. 420 for 7 minutes)."
    )
    denoise: Optional[float] = Field(
        default=0.75,
        description="Optional: Denoise level for V2V (0.50 to 0.88, default 0.75)."
    )
    control_mode: Optional[str] = Field(
        default=None,
        description="Control video mode: 'pose+depth' (default), 'pose', 'depth', or 'canny'."
    )
    cloning_backend: Optional[str] = Field(
        default=None,
        description="Cloning backend: 'fun_control' (default) or 'legacy_i2v'."
    )
    prompt_override: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional prompt overrides: 'fun_control_positive', 'fun_control_negative', 'qwen_keyframe', 'qwen_negative'"
    )
    job_id: Optional[str] = Field(
        default=None,
        description="Optional existing job_id to resume without creating a new folder."
    )


class ResumeJobRequest(BaseModel):
    job_id: str = Field(description="The existing job_id to resume")
    max_duration: Optional[float] = None
    denoise: Optional[float] = None
    control_mode: Optional[str] = None
    cloning_backend: Optional[str] = None
    prompt_override: Optional[Dict[str, str]] = None


def resolve_input(name: str) -> Path:
    input_dir = CLONING_INPUT_DIR.resolve()
    base_input = (BASE_DIR / "input").resolve()

    clean_name = name.strip()
    file_basename = Path(clean_name).name

    candidates = [
        Path(clean_name),
        BASE_DIR / clean_name,
        input_dir / clean_name,
        input_dir / file_basename,
        base_input / clean_name,
        base_input / file_basename,
    ]

    path = None
    for cand in candidates:
        try:
            cand_res = cand.resolve()
            if cand_res.is_file():
                path = cand_res
                break
        except Exception:
            continue

    if path is None:
        raise HTTPException(
            status_code=422,
            detail=f"File not found in input/cloning or input: {name}",
        )

    if path.stat().st_size == 0:
        raise HTTPException(
            status_code=422,
            detail=f"File '{name}' is 0 bytes (empty or still uploading). Please wait for upload to finish.",
        )

    # Validate that MP4 has finished uploading and has valid container/moov atom
    probe_cmd = ["ffprobe", "-v", "error", str(path)]
    probe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if probe_res.returncode != 0:
        err_msg = probe_res.stderr.strip()
        if "moov atom not found" in err_msg:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"File '{name}' ({path.stat().st_size / (1024*1024):.1f} MB) is incomplete or still uploading "
                    f"in JupyterLab (moov atom not found). Please wait until JupyterLab finishes writing 100% of the file."
                ),
            )
        raise HTTPException(
            status_code=422,
            detail=f"File '{name}' is unreadable or corrupted: {err_msg}",
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
    max_duration: Optional[float] = None,
    denoise: Optional[float] = 0.75,
    control_mode: Optional[str] = None,
    cloning_backend: Optional[str] = None,
    prompt_override: Optional[Dict[str, str]] = None,
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
            char_meta = []

            if characters:
                for c in characters:
                    # On clonin-anime branch, strictly enforce anime conversion
                    style = "anime"
                    c_gender = (c.gender or "character").lower().strip()
                    c_age = c.age or 22

                    if c.character_prompt and c.character_prompt.strip():
                        p = c.character_prompt.strip()
                        # Sanitize any leftover photorealistic keywords from client or postman payloads
                        realistic_keywords = [
                            "photorealistic", "photograph", "raw photo", "35mm film",
                            "natural skin pores", "skin pores", "skin texture", "authentic human",
                            "human skin", "realistic facial anatomy", "hyperrealistic", "realistic"
                        ]
                        for kw in realistic_keywords:
                            p = p.replace(kw, "")
                        p_clean = " ".join(p.split()).strip(" ,.")
                        p = f"masterpiece 2D Japanese anime {c_gender}, Makoto Shinkai aesthetic, clean cel-shaded anime linework, expressive anime eyes, {p_clean}"
                    else:
                        p = f"gorgeous attractive adult 2D anime {c_gender} with expressive detailed anime eyes, stylized anime hair, radiant smooth anime skin, clean cel-shaded aesthetic"
                    prompts.append(p)
                    char_meta.append({
                        "gender": c_gender,
                        "age": c_age,
                        "character_prompt": p,
                        "style": "anime",
                    })
            else:
                # Direct Video-to-Video mode: No characters required.
                # The video itself provides all actors, motion, choreography, and scene structure.
                logger.info("[WAN CLONING] No character list provided. Running pure Video-to-Video (V2V) transformation directly on video frames.")
                prompts = []
                char_meta = []

            init_status = f"generating_{style}_keyframe" if prompts else "animating_with_wan"
            write_status(job_dir, job_id=job_id, status=init_status)

            def chunk_status_cb(**kwargs):
                write_status(job_dir, job_id=job_id, **kwargs)

            # Step 2: Animate with Wan 2.2 V2V
            write_status(job_dir, job_id=job_id, status="animating_with_wan")
            output_video_path = job_dir / "cloned_output.mp4"

            logger.info(f"[WAN CLONING] Running Wan 2.2 Cloner ({style}) on {source_video}")
            cloner.clone_video(
                source_video=source_video,
                character_prompts=prompts,
                output_video=output_video_path,
                job_dir=job_dir,
                seed=int(uuid.uuid4().int % 1000000),
                style=style,
                characters=char_meta,
                status_callback=chunk_status_cb,
                max_duration=max_duration,
                denoise=denoise or 0.75,
                cloning_backend=cloning_backend,
                control_mode=control_mode,
                prompt_override=prompt_override,
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
                model_used="Wan 2.2 V2V 14B Dual Experts + Lightning LoRA",
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
    target_video_str = request.source_video or request.video_path
    if not target_video_str:
        raise HTTPException(
            status_code=422,
            detail="Either 'source_video' or 'video_path' must be provided in the request body.",
        )
    source_video = resolve_input(target_video_str)

    if request.job_id and request.job_id.strip():
        job_id = request.job_id.strip()
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[CLONING API] Resuming existing job_id: {job_id}")
    else:
        job_id = f"clone_{uuid.uuid4().hex[:12]}"
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the input source video directly into the job's output directory
    input_video_dest = job_dir / f"input_{source_video.name}"
    if not input_video_dest.exists() or input_video_dest.stat().st_size == 0:
        try:
            shutil.copy2(source_video, input_video_dest)
            active_source_video = input_video_dest
        except Exception as copy_err:
            logger.warning(f"Could not copy source video into {job_dir}: {copy_err}")
            active_source_video = source_video
    else:
        active_source_video = input_video_dest

    with open(job_dir / "cloning_record.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_video": target_video_str,
                "input_video_copy": str(active_source_video),
                "characters": [c.model_dump() for c in request.characters] if request.characters else [],
                "prompt_override": request.prompt_override,
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
        request.mode or "wan_v2v",
        request.max_duration,
        request.denoise or 0.75,
        request.control_mode,
        request.cloning_backend,
        request.prompt_override,
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
    max_duration: Optional[float] = Form(
        default=None,
        description="Optional: Max seconds of video to process (defaults to full video, e.g. 420 for 7 minutes)."
    ),
    denoise: Optional[float] = Form(
        default=0.75,
        description="Optional: Denoise level for V2V (0.50 to 0.88, default 0.75)."
    ),
    control_mode: Optional[str] = Form(
        default=None,
        description="Control video mode: 'pose+depth' (default), 'pose', 'depth', or 'canny'."
    ),
    cloning_backend: Optional[str] = Form(
        default=None,
        description="Cloning backend: 'fun_control' (default) or 'legacy_i2v'."
    ),
    prompt_override: Optional[str] = Form(
        default=None,
        description="Optional JSON string of prompt overrides: e.g. {'fun_control_positive': '...', 'fun_control_negative': '...'}"
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

    parsed_prompt_override = None
    if prompt_override and prompt_override.strip():
        try:
            parsed_prompt_override = json.loads(prompt_override)
            if not isinstance(parsed_prompt_override, dict):
                parsed_prompt_override = None
        except Exception as err:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid 'prompt_override' JSON form field: {err}"
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

    active_mode = (mode or "wan_v2v").lower().strip()

    with open(job_dir / "cloning_record.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_video": safe_name,
                "input_video_copy": str(saved_video_path),
                "characters": [c.model_dump() for c in char_targets],
                "mode": active_mode,
                "control_mode": control_mode,
                "cloning_backend": cloning_backend,
                "prompt_override": parsed_prompt_override,
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
        char_targets if char_targets else None,
        active_mode,
        max_duration,
        denoise or 0.75,
        control_mode,
        cloning_backend,
        parsed_prompt_override,
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


@router.post("/resume")
def resume_cloning_job(request: ResumeJobRequest):
    """Resumes an interrupted cloning job using its existing directory and saved settings."""
    return _resume_job_by_id(
        job_id=request.job_id,
        max_duration=request.max_duration,
        denoise=request.denoise,
        control_mode=request.control_mode,
        cloning_backend=request.cloning_backend,
        prompt_override=request.prompt_override,
    )


@router.post("/jobs/{job_id}/resume")
def resume_cloning_job_by_path(
    job_id: str,
    max_duration: Optional[float] = None,
    denoise: Optional[float] = None,
    control_mode: Optional[str] = None,
    cloning_backend: Optional[str] = None,
    prompt_override: Optional[str] = None,
):
    """Resumes an interrupted cloning job by job_id URL path parameter."""
    parsed_override = None
    if prompt_override:
        try:
            parsed_override = json.loads(prompt_override)
        except Exception:
            pass
    return _resume_job_by_id(
        job_id=job_id,
        max_duration=max_duration,
        denoise=denoise,
        control_mode=control_mode,
        cloning_backend=cloning_backend,
        prompt_override=parsed_override,
    )


def _resume_job_by_id(
    job_id: str,
    max_duration: Optional[float] = None,
    denoise: Optional[float] = None,
    control_mode: Optional[str] = None,
    cloning_backend: Optional[str] = None,
    prompt_override: Optional[Dict[str, str]] = None,
):
    clean_id = job_id.strip()
    job_dir = OUTPUT_DIR / clean_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job directory not found for ID '{clean_id}'")

    state_file = job_dir / "job_state.json"
    record_file = job_dir / "cloning_record.json"

    saved_state = {}
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as sf:
                saved_state = json.load(sf)
        except Exception as se:
            logger.warning(f"Could not read {state_file}: {se}")

    record_data = {}
    if record_file.exists():
        try:
            with open(record_file, "r", encoding="utf-8") as rf:
                record_data = json.load(rf)
        except Exception as re:
            logger.warning(f"Could not read {record_file}: {re}")

    source_video_path = None
    input_copies = list(job_dir.glob("input_*.*"))
    if input_copies and input_copies[0].is_file() and input_copies[0].stat().st_size > 0:
        source_video_path = input_copies[0]
    elif "input_video_copy" in record_data and Path(record_data["input_video_copy"]).is_file():
        source_video_path = Path(record_data["input_video_copy"])
    elif "source_video" in saved_state and Path(saved_state["source_video"]).is_file():
        source_video_path = Path(saved_state["source_video"])
    elif "source_video" in record_data:
        try:
            source_video_path = resolve_input(record_data["source_video"])
        except Exception:
            pass

    if source_video_path is None or not source_video_path.exists():
        raise HTTPException(
            status_code=422,
            detail=f"Could not locate source video in {job_dir} to resume job.",
        )

    raw_characters = record_data.get("characters") or []
    characters = [CharacterTarget(**c) for c in raw_characters] if raw_characters else None

    active_backend = cloning_backend or saved_state.get("cloning_backend") or "fun_control"
    active_control = control_mode or saved_state.get("control_mode") or "pose+depth"
    active_denoise = denoise if denoise is not None else saved_state.get("denoise", 0.75)
    active_max_dur = max_duration if max_duration is not None else saved_state.get("active_duration")
    active_prompt_override = (
        prompt_override
        or saved_state.get("prompt_override")
        or record_data.get("prompt_override")
    )

    write_status(
        job_dir,
        job_id=clean_id,
        status="resuming",
        input_video_url=f"/output/{clean_id}/{source_video_path.name}",
    )

    executor.submit(
        process_cloning_job,
        clean_id,
        job_dir,
        source_video_path,
        characters,
        "wan_v2v",
        active_max_dur,
        active_denoise,
        active_control,
        active_backend,
        active_prompt_override,
    )

    return {
        "job_id": clean_id,
        "status": "resumed",
        "resumed_from": str(job_dir),
        "status_url": f"/api/v1/cloning/jobs/{clean_id}",
    }
