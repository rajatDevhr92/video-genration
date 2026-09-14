import argparse
import json
import os
import time
import uuid

import requests


COMFY_URL = "http://127.0.0.1:8188"
WORKFLOW = "/workspace/story-video-engine/workflows/qwen_image_2512_api.json"


def submit(prompt, width, height, seed):
    with open(WORKFLOW, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    workflow["5"]["inputs"]["text"] = prompt
    workflow["7"]["inputs"]["width"] = width
    workflow["7"]["inputs"]["height"] = height
    workflow["8"]["inputs"]["seed"] = seed

    client_id = str(uuid.uuid4())

    response = requests.post(
        f"{COMFY_URL}/prompt",
        json={
            "prompt": workflow,
            "client_id": client_id
        },
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(json.dumps(data, indent=2))

    return data["prompt_id"]


def wait_for_result(prompt_id):
    start = time.time()

    while True:
        response = requests.get(
            f"{COMFY_URL}/history/{prompt_id}",
            timeout=30
        )

        response.raise_for_status()

        history = response.json()

        if prompt_id in history:
            result = history[prompt_id]

            if result.get("status", {}).get("status_str") == "error":
                raise RuntimeError(
                    json.dumps(result, indent=2)
                )

            if result.get("outputs"):
                return result, time.time() - start

        elapsed = int(time.time() - start)

        print(f"Waiting for Qwen generation... {elapsed}s")

        time.sleep(2)


def download_image(result, output_path):
    outputs = result.get("outputs", {})

    for node_id, node_output in outputs.items():
        for image in node_output.get("images", []):
            filename = image["filename"]
            subfolder = image.get("subfolder", "")
            folder_type = image.get("type", "output")

            response = requests.get(
                f"{COMFY_URL}/view",
                params={
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": folder_type
                },
                timeout=60
            )

            response.raise_for_status()

            os.makedirs(
                os.path.dirname(output_path),
                exist_ok=True
            )

            with open(output_path, "wb") as f:
                f.write(response.content)

            return output_path

    raise RuntimeError("No image found in ComfyUI output")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        required=True
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=464857551335368
    )

    parser.add_argument(
        "--output",
        default="/workspace/story-video-engine/output/test_qwen.png"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("QWEN IMAGE 2512 TEST")
    print("=" * 80)

    print("Prompt:", args.prompt)
    print("Resolution:", f"{args.width}x{args.height}")
    print("Seed:", args.seed)

    print("\nSubmitting workflow...")

    prompt_id = submit(
        args.prompt,
        args.width,
        args.height,
        args.seed
    )

    print("Prompt ID:", prompt_id)
    print("Generation started.\n")

    result, generation_time = wait_for_result(prompt_id)

    output = download_image(
        result,
        args.output
    )

    print("\n" + "=" * 80)
    print("GENERATION COMPLETED")
    print("=" * 80)

    print(f"Generation time: {generation_time:.2f}s")
    print("Image:", output)


if __name__ == "__main__":
    main()