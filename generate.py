#!/usr/bin/env python3

import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

from app.comfy_client import ComfyClient


BASE_DIR = Path(__file__).resolve().parent

WORKFLOW_PATH = (
    BASE_DIR
    / "workflows"
    / "wan2.2_t2v_14b_api.json"
)

OUTPUT_ROOT = BASE_DIR / "output"


def build_workflow(
    prompt,
    width,
    height,
    duration,
    fps,
    job_id,
):
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # Positive prompt
    workflow["128:89"]["inputs"]["text"] = prompt

    # Resolution
    workflow["128:74"]["inputs"]["width"] = width
    workflow["128:74"]["inputs"]["height"] = height

    # Duration
    workflow["128:126"]["inputs"]["value"] = duration

    # FPS
    workflow["128:125"]["inputs"]["value"] = fps

    # Wan 2.2 Lightning acceleration
    workflow["128:129"]["inputs"]["value"] = True

    # Unique ComfyUI output prefix
    workflow["80"]["inputs"]["filename_prefix"] = (
        f"video/story_{job_id}"
    )

    return workflow


def save_metadata(job_dir, metadata):
    metadata_path = job_dir / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return metadata_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Wan 2.2 video through ComfyUI"
    )

    parser.add_argument(
        "--prompt",
        required=True,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=5,
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=16,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Job
    # ---------------------------------------------------------

    job_id = uuid.uuid4().hex[:12]

    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    metadata = {
        "job_id": job_id,
        "status": "created",
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "duration": args.duration,
        "fps": args.fps,
        "lightning": True,
        "created_at": time.time(),
    }

    save_metadata(job_dir, metadata)

    print("=" * 70)
    print("WAN 2.2 VIDEO GENERATOR")
    print("=" * 70)

    print(f"Job ID:     {job_id}")
    print(f"Prompt:     {args.prompt}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Duration:   {args.duration}s")
    print(f"FPS:        {args.fps}")
    print("Lightning:  ON")
    print()

    workflow = build_workflow(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        duration=args.duration,
        fps=args.fps,
        job_id=job_id,
    )

    client = ComfyClient(
        base_url="http://127.0.0.1:8188",
        output_dir="/workspace/ComfyUI/output",
    )

    # ---------------------------------------------------------
    # Submit
    # ---------------------------------------------------------

    print("Submitting workflow to ComfyUI...")

    started = time.time()

    try:
        prompt_id, client_id = client.submit(workflow)

    except Exception as e:
        metadata["status"] = "failed"
        metadata["error"] = str(e)
        metadata["failed_at"] = time.time()
        save_metadata(job_dir, metadata)

        print("SUBMISSION FAILED")
        print(str(e))

        sys.exit(1)

    metadata["status"] = "processing"
    metadata["prompt_id"] = prompt_id
    metadata["client_id"] = client_id

    save_metadata(job_dir, metadata)

    print(f"ComfyUI Prompt ID: {prompt_id}")
    print(f"Client ID:         {client_id}")
    print()
    print("Generation started.")
    print()

    # ---------------------------------------------------------
    # Wait
    # ---------------------------------------------------------

    try:
        files = client.wait_for_completion(
            prompt_id=prompt_id,
            timeout=args.timeout,
        )

    except Exception as e:
        elapsed = time.time() - started

        metadata["status"] = "failed"
        metadata["error"] = str(e)
        metadata["generation_time_seconds"] = round(
            elapsed,
            2,
        )
        metadata["failed_at"] = time.time()

        save_metadata(job_dir, metadata)

        print()
        print("GENERATION FAILED")
        print(str(e))

        sys.exit(1)

    elapsed = time.time() - started

    # ---------------------------------------------------------
    # Find video output
    # ---------------------------------------------------------

    video_files = [
        item
        for item in files
        if item.get("filename", "").lower().endswith(
            (".mp4", ".webm", ".mov", ".mkv")
        )
    ]

    if not video_files:
        metadata["status"] = "failed"
        metadata["error"] = (
            "ComfyUI completed but no video output was found."
        )
        metadata["generation_time_seconds"] = round(
            elapsed,
            2,
        )

        save_metadata(job_dir, metadata)

        print()
        print("GENERATION FAILED")
        print("No video output found.")

        sys.exit(1)

    # We use the first video returned by this specific prompt.
    source_video = Path(video_files[0]["path"])

    if not source_video.exists():
        metadata["status"] = "failed"
        metadata["error"] = (
            f"Generated video does not exist: {source_video}"
        )

        save_metadata(job_dir, metadata)

        print()
        print("GENERATION FAILED")
        print(metadata["error"])

        sys.exit(1)

    # ---------------------------------------------------------
    # Copy into engine-owned job directory
    # ---------------------------------------------------------

    destination_video = job_dir / "video.mp4"

    shutil.copy2(
        source_video,
        destination_video,
    )

    # ---------------------------------------------------------
    # Final metadata
    # ---------------------------------------------------------

    metadata["status"] = "completed"
    metadata["generation_time_seconds"] = round(
        elapsed,
        2,
    )
    metadata["source_video"] = str(source_video)
    metadata["video"] = str(destination_video)
    metadata["video_size_bytes"] = destination_video.stat().st_size
    metadata["completed_at"] = time.time()

    save_metadata(job_dir, metadata)

    # ---------------------------------------------------------
    # Result
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATION COMPLETED")
    print("=" * 70)

    print(f"Job ID:          {job_id}")
    print(f"Generation time: {elapsed:.2f} seconds")
    print()
    print(f"Video:")
    print(f"  {destination_video}")
    print()
    print(f"Metadata:")
    print(f"  {job_dir / 'metadata.json'}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()