import json
import os
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.local_llm_provider import LocalLLMProvider
from app.story_planner import (
    StoryPlanner,
    StoryPlanValidationError,
)
from app.job_cleanup import cleanup_completed_job
from app.gpu_memory import prepare_for_local_llm


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
OUTPUT_DIR = BASE_DIR / "output"

HOST = os.getenv(
    "STORY_API_HOST",
    "0.0.0.0",
)

PORT = int(
    os.getenv(
        "STORY_API_PORT",
        "8000",
    )
)

# Important:
# Only one generation runs at a time because all jobs share
# the same H200 / ComfyUI instance.
executor = ThreadPoolExecutor(
    max_workers=1
)

jobs = {}
jobs_lock = threading.Lock()


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Story Video Engine",
    version="1.0.0",
)


# Serve generated files.
app.mount(
    "/outputs",
    StaticFiles(
        directory=str(OUTPUT_DIR)
    ),
    name="outputs",
)


# ============================================================
# Request model
# ============================================================

class StoryRequest(BaseModel):
    story: str = Field(
        ...,
        min_length=10,
        description="Complete story to turn into a cinematic video.",
    )
    target_scenes: int | None = Field(
        default=None,
        description="Optional target scene count for the story planner.",
    )


# ============================================================
# Job helpers
# ============================================================

def update_job(job_id, **values):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(values)


def get_job(job_id):
    with jobs_lock:
        return jobs.get(job_id)


def run_command(command):
    result = subprocess.run(
        command,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n"
            f"{' '.join(command)}\n\n"
            f"{result.stdout}"
        )

    return result.stdout


# ============================================================
# Complete story pipeline
# ============================================================

def generate_story_job(
    job_id,
    story,
    target_scenes=None,
):
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        job_dir / "generation.log"
    )

    def log(message):
        print(
            f"[{job_id}] {message}"
        )

        with open(
            log_path,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                message + "\n"
            )

    try:
        # ----------------------------------------------------
        # 1. Story planning
        # ----------------------------------------------------

        update_job(
            job_id,
            status="planning",
        )

        log("Creating story plan...")

        if os.getenv("OPENAI_API_KEY"):
            from app.openai_provider import OpenAIProvider
            provider = OpenAIProvider()
        else:
            provider = LocalLLMProvider()

        planner = StoryPlanner(
            provider=provider
        )

        plan = planner.build(
            story,
            target_scenes=target_scenes,
        )

        plan_path = (
            job_dir
            / "story_plan.json"
        )

        with open(
            plan_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                plan,
                f,
                indent=2,
                ensure_ascii=False,
            )

        scenes = plan.get(
            "scenes",
            [],
        )

        if not scenes:
            raise RuntimeError(
                "Story planner returned zero scenes."
            )

        scene_count = len(
            scenes
        )

        total_duration = sum(
            float(
                scene.get(
                    "duration",
                    5,
                )
            )
            for scene in scenes
        )

        log(
            f"Plan created: "
            f"{scene_count} scenes / "
            f"{total_duration:.1f}s"
        )

        update_job(
            job_id,
            status="generating_video",
            scenes=scene_count,
            duration_seconds=total_duration,
        )

        # ----------------------------------------------------
        # 2. Generate all scenes
        # ----------------------------------------------------

        log(
            "Generating characters, keyframes "
            "and Wan 2.2 videos..."
        )

        run_command([
            "python",
            "generate_story.py",
            "--job-id",
            job_id,
            "--story-plan",
            str(plan_path),
            "--start-scene",
            "1",
            "--end-scene",
            str(scene_count),
        ])

        # ----------------------------------------------------
        # 3. Generate speech audio
        # ----------------------------------------------------

        update_job(
            job_id,
            status="generating_audio",
        )

        log(
            "Generating narration/dialogue audio..."
        )

        run_command([
            "python",
            "add_audio_to_story.py",
            str(job_dir),
        ])

        # ----------------------------------------------------
        # 4. Assemble final video
        # ----------------------------------------------------

        update_job(
            job_id,
            status="assembling",
        )

        log(
            "Assembling final video..."
        )

        # Import here so the API process always uses
        # the current implementation.
        from app.video_assembler import (
            VideoAssembler,
        )

        final_path = (
            job_dir
            / "final"
            / "final_with_audio.mp4"
        )

        result = VideoAssembler.assemble(
            job_dir=job_dir,
            output_path=final_path,
        )

        # cleanup_completed_job(
        #     job_dir=job_dir,
        #     final_video_name="final_with_audio.mp4",
        # )

        # ----------------------------------------------------
        # 5. Completed
        # ----------------------------------------------------

        relative_video = (
            final_path.relative_to(
                OUTPUT_DIR
            )
        )

        update_job(
            job_id,
            status="completed",
            output=str(final_path),
            video_url=(
                f"/outputs/"
                f"{relative_video.as_posix()}"
            ),
            scene_count=scene_count,
            duration_seconds=total_duration,
        )

        log(
            f"Completed: {final_path}"
        )

    except StoryPlanValidationError as exc:

        log(
            "STORY PLAN INVALID: "
            + exc.message
        )

        update_job(
            job_id,
            status="failed",
            error={
                "code": "STORY_PLAN_INVALID",
                "message": exc.message,
                "missing_fields": (
                    exc.missing_fields
                ),
                "details": exc.details,
            },
        )

    except Exception as exc:

        log(
            "FAILED:\n"
            + repr(exc)
        )

        update_job(
            job_id,
            status="failed",
            error={
                "code": "STORY_GENERATION_FAILED",
                "message": str(exc),
            },
        )


# ============================================================
# API endpoints
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "story-video-engine",
    }


@app.post("/api/v1/stories")
def create_story(request: StoryRequest):

    story = request.story.strip()

    if not story:
        raise HTTPException(
            status_code=422,
            detail="Story cannot be empty.",
        )

    job_id = (
        f"story_"
        f"{uuid.uuid4().hex[:12]}"
    )

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "story": story,
        }

    executor.submit(
        generate_story_job,
        job_id,
        story,
        request.target_scenes,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": (
            "Story accepted. "
            "Generation has started."
        ),
        "status_url": (
            f"/api/v1/stories/{job_id}"
        ),
    }


@app.get("/api/v1/stories/{job_id}")
def story_status(job_id: str):

    job = get_job(
        job_id
    )

    if not job:

        # Allow looking up completed jobs after
        # an API restart if the output exists.
        job_dir = (
            OUTPUT_DIR / job_id
        )

        final_path = (
            job_dir
            / "final"
            / "final_with_audio.mp4"
        )

        if final_path.exists():
            return {
                "job_id": job_id,
                "status": "completed",
                "output": str(
                    final_path
                ),
                "video_url": (
                    f"/outputs/"
                    f"{job_id}/final/"
                    f"final_with_audio.mp4"
                ),
            }

        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    return job