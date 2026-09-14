import gc
import os
import time

import requests
import torch


COMFY_URL = os.getenv(
    "COMFY_URL",
    "http://127.0.0.1:8188",
)


def cleanup_python_gpu():
    """
    Release memory owned by the current Python process.
    """
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            print(
                f"[MEMORY] CUDA cache cleanup warning: {e}"
            )


def request_comfy_memory_release(
    wait_seconds=2,
):
    """
    Ask ComfyUI to unload its models and release cached memory.

    ComfyUI's /free endpoint schedules the cleanup; it does not
    guarantee that VRAM is already free when the HTTP response arrives.
    """

    print(
        "[MEMORY] Requesting ComfyUI to unload models..."
    )

    try:
        response = requests.post(
            f"{COMFY_URL}/free",
            json={
                "unload_models": True,
                "free_memory": True,
            },
            timeout=30,
        )

        if response.status_code != 200:
            print(
                "[MEMORY] ComfyUI /free returned "
                f"HTTP {response.status_code}: "
                f"{response.text}"
            )
            return False

        print(
            "[MEMORY] ComfyUI memory release requested."
        )

    except Exception as e:
        print(
            "[MEMORY] Could not contact ComfyUI: "
            f"{e}"
        )
        return False

    if wait_seconds > 0:
        time.sleep(wait_seconds)

    cleanup_python_gpu()

    return True


def print_gpu_memory(label="GPU"):
    """
    Print current CUDA memory statistics.
    """
    if not torch.cuda.is_available():
        print(
            f"[MEMORY] {label}: CUDA unavailable"
        )
        return

    try:
        device = torch.cuda.current_device()

        allocated = (
            torch.cuda.memory_allocated(device)
            / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved(device)
            / (1024 ** 3)
        )

        total = (
            torch.cuda.get_device_properties(device)
            .total_memory
            / (1024 ** 3)
        )

        print(
            f"[MEMORY] {label}: "
            f"allocated={allocated:.2f} GB, "
            f"reserved={reserved:.2f} GB, "
            f"total={total:.2f} GB"
        )

    except Exception as e:
        print(
            f"[MEMORY] Could not read GPU memory: {e}"
        )


def prepare_for_local_llm():
    """
    Prepare the GPU for the local Transformer LLM.

    ComfyUI must release its loaded image/video models first.
    """
    print()
    print("=" * 70)
    print("GPU MEMORY HANDOFF: COMFYUI -> LOCAL LLM")
    print("=" * 70)

    print_gpu_memory(
        "Before ComfyUI release"
    )

    request_comfy_memory_release(
        wait_seconds=3
    )

    print_gpu_memory(
        "After ComfyUI release"
    )


def prepare_for_comfy():
    """
    Prepare the GPU for ComfyUI.

    The local LLM should already have been unloaded by this point.
    """
    print()
    print("=" * 70)
    print("GPU MEMORY HANDOFF: LOCAL LLM -> COMFYUI")
    print("=" * 70)

    cleanup_python_gpu()

    print_gpu_memory(
        "Before ComfyUI generation"
    )