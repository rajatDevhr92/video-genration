"""
cloning_routes.py - separate API routes for cloning jobs.

This file does not use the story planner, anatomy or explicit prompt code.
Generation is not connected yet: a job that passes review stops at
"approved_awaiting_generation".
"""
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import BASE_DIR, OUTPUT_DIR

from .cloning_record import CloningRecord
from .cloning_safety import review_media

CLONING_INPUT_DIR = BASE_DIR / "input" / "cloning"

router = APIRouter(
    prefix="/api/v1/cloning",
    tags=["cloning"],
)

# The safety models share the server's GPU, so review one job at a time.
executor = ThreadPoolExecutor(max_workers=1)


class CloningJobRequest(BaseModel):
    source_video: str = Field(
        ...,
        description="File name inside input/cloning on the server.",
    )
    character_image: str = Field(
        ...,
        description="File name inside input/cloning on the server.",
    )
    record: CloningRecord


def resolve_input(name):
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


def write_status(job_dir, **values):
    with open(
        job_dir / "cloning_status.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            values,
            f,
            indent=2,
            ensure_ascii=False,
        )


def run_review(job_id, job_dir, source_video, character_image):
    write_status(
        job_dir,
        job_id=job_id,
        status="reviewing",
    )

    try:
        report = review_media(
            source_video,
            character_image,
        )
    except Exception as error:
        report = {
            "passed": False,
            "reasons": [f"safety check could not run: {error}"],
            "frames_checked": 0,
        }

    write_status(
        job_dir,
        job_id=job_id,
        status=(
            "approved_awaiting_generation"
            if report["passed"]
            else "rejected"
        ),
        review=report,
    )


@router.post("/jobs")
def create_cloning_job(request: CloningJobRequest):
    source_video = resolve_input(request.source_video)
    character_image = resolve_input(request.character_image)

    job_id = f"clone_{uuid.uuid4().hex[:12]}"
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        job_dir / "cloning_record.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "source_video": request.source_video,
                "character_image": request.character_image,
                **request.record.model_dump(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    write_status(
        job_dir,
        job_id=job_id,
        status="queued",
    )

    executor.submit(
        run_review,
        job_id,
        job_dir,
        source_video,
        character_image,
    )

    return {
        "job_id": job_id,
        "status": "queued",
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

    with open(
        status_path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)
