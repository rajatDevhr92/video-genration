#!/usr/bin/env python3

import argparse
import json
import shutil
import time
import uuid
from pathlib import Path

import requests


COMFY_URL = "http://127.0.0.1:8188"

BASE_DIR = Path("/workspace/story-video-engine")

WORKFLOW_PATH = (
    BASE_DIR
    / "workflows"
    / "qwen_image_edit_2511_api.json"
)

COMFY_INPUT_DIR = Path("/workspace/ComfyUI/input")


def load_workflow():
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def submit_workflow(workflow):
    client_id = str(uuid.uuid4())

    response = requests.post(
        f"{COMFY_URL}/prompt",
        json={
            "prompt": workflow,
            "client_id": client_id,
        },
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(
            "ComfyUI rejected workflow:\n"
            + json.dumps(data, indent=2)
        )

    prompt_id = data.get("prompt_id")

    if not prompt_id:
        raise RuntimeError(
            "ComfyUI did not return prompt_id:\n"
            + json.dumps(data, indent=2)
        )

    return prompt_id


def wait_for_result(prompt_id, timeout=1800):
    started = time.time()

    while True:
        elapsed = time.time() - started

        if elapsed > timeout:
            raise TimeoutError(
                f"Qwen Image Edit timed out after {timeout}s"
            )

        response = requests.get(
            f"{COMFY_URL}/history/{prompt_id}",
            timeout=30,
        )

        response.raise_for_status()

        history = response.json()

        if prompt_id not in history:
            print(
                f"Waiting for Qwen keyframe... "
                f"{int(elapsed)}s",
                flush=True,
            )
            time.sleep(2)
            continue

        result = history[prompt_id]

        status = result.get("status", {})

        if status.get("status_str") == "error":
            raise RuntimeError(
                "Qwen Image Edit failed:\n"
                + json.dumps(result, indent=2)
            )

        outputs = result.get("outputs", {})

        if outputs:
            return result, elapsed

        print(
            f"Processing... {int(elapsed)}s",
            flush=True,
        )

        time.sleep(2)


def download_result(result, output_path):
    outputs = result.get("outputs", {})

    for node_id, node_output in outputs.items():

        for image in node_output.get("images", []):

            filename = image.get("filename")

            if not filename:
                continue

            subfolder = image.get("subfolder", "")
            file_type = image.get("type", "output")

            response = requests.get(
                f"{COMFY_URL}/view",
                params={
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": file_type,
                },
                timeout=120,
            )

            response.raise_for_status()

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path.write_bytes(response.content)

            return {
                "filename": filename,
                "subfolder": subfolder,
                "type": file_type,
                "path": str(output_path),
                "source_node": node_id,
            }

    raise RuntimeError(
        "ComfyUI completed but no image output was found."
    )


def prepare_workflow(
    reference_path,
    prompt,
    seed,
):
    workflow = load_workflow()

    reference_path = Path(reference_path)

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference image not found: {reference_path}"
        )

    COMFY_INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ComfyUI LoadImage expects a file available in its input folder.
    reference_filename = (
        f"story_reference_{uuid.uuid4().hex[:10]}"
        f"{reference_path.suffix.lower()}"
    )

    comfy_reference = (
        COMFY_INPUT_DIR
        / reference_filename
    )

    shutil.copy2(
        reference_path,
        comfy_reference,
    )

    # Primary reference image
    workflow["1000"]["inputs"]["image"] = (
        reference_filename
    )

    # Negative/reference-preservation instruction
    workflow["9"]["inputs"]["prompt"] = (
        "Preserve the identity, facial structure, hairstyle, "
        "body proportions and clothing of the reference character. "
        "Do not redesign the character."
    )

    # Actual scene prompt
    workflow["13"]["inputs"]["prompt"] = prompt

    # Seed
    workflow["15"]["inputs"]["seed"] = seed

    # Unique output prefix
    job_id = uuid.uuid4().hex[:12]

    workflow["1001"]["inputs"]["filename_prefix"] = (
        f"story_keyframes/{job_id}"
    )

    return workflow, job_id, comfy_reference


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Qwen Image Edit 2511 scene keyframe."
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Master character/reference image",
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Scene keyframe prompt",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Final keyframe output path",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123456789,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    print("=" * 80)
    print("QWEN IMAGE EDIT 2511 — KEYFRAME GENERATOR")
    print("=" * 80)
    print()
    print("Reference:", args.reference)
    print("Output:   ", args.output)
    print("Seed:     ", args.seed)
    print()
    print("Scene prompt:")
    print(args.prompt)
    print()

    workflow, job_id, comfy_reference = prepare_workflow(
        reference_path=args.reference,
        prompt=args.prompt,
        seed=args.seed,
    )

    print("Job ID:", job_id)
    print()
    print("Submitting to ComfyUI...")

    started = time.time()

    try:
        prompt_id = submit_workflow(workflow)

        print("Prompt ID:", prompt_id)
        print()
        print("Generation started.")
        print()

        result, elapsed = wait_for_result(
            prompt_id,
            timeout=args.timeout,
        )

        saved = download_result(
            result,
            output_path,
        )

    finally:
        # Remove temporary copy from ComfyUI input.
        try:
            comfy_reference.unlink(missing_ok=True)
        except Exception:
            pass

    total_time = time.time() - started

    print()
    print("=" * 80)
    print("KEYFRAME GENERATION COMPLETED")
    print("=" * 80)
    print()
    print("Generation time:", f"{total_time:.2f}s")
    print("Output:", saved["path"])
    print("ComfyUI file:", saved["filename"])
    print()


if __name__ == "__main__":
    main()