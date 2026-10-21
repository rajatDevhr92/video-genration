import gc
import os
import time

import requests

try:
    import torch
except ImportError:
    torch = None


COMFY_URL = os.getenv(
    "COMFY_URL",
    "http://127.0.0.1:8188",
)

DEFAULT_MIN_FREE_GPU_MEMORY_GB = float(
    os.getenv("MIN_FREE_GPU_MEMORY_GB", "2.0")
)


def get_free_gpu_memory(device=None):
    """
    Return (free_bytes, total_bytes) using torch.cuda.mem_get_info().
    """
    if torch is None or not torch.cuda.is_available():
        return 0, 0
    return torch.cuda.mem_get_info(device)


def check_free_gpu_memory(
    min_free_gb=None,
    min_free_bytes=None,
    device=None,
):
    """
    Check free GPU memory with torch.cuda.mem_get_info().
    Logs the free memory. If free memory is below the configurable minimum,
    raises a RuntimeError.
    """
    if torch is None or not torch.cuda.is_available():
        print("[MEMORY] GPU memory check: CUDA unavailable")
        return 0, 0

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)

    if min_free_bytes is not None:
        min_bytes = int(min_free_bytes)
        min_limit_gb = min_bytes / (1024 ** 3)
    elif min_free_gb is not None:
        min_limit_gb = float(min_free_gb)
        min_bytes = int(min_limit_gb * (1024 ** 3))
    elif os.getenv("MIN_FREE_GPU_MEMORY_GB") is not None:
        min_limit_gb = float(os.getenv("MIN_FREE_GPU_MEMORY_GB"))
        min_bytes = int(min_limit_gb * (1024 ** 3))
    elif os.getenv("MIN_FREE_GPU_MEMORY_BYTES") is not None:
        min_bytes = int(os.getenv("MIN_FREE_GPU_MEMORY_BYTES"))
        min_limit_gb = min_bytes / (1024 ** 3)
    else:
        min_limit_gb = DEFAULT_MIN_FREE_GPU_MEMORY_GB
        min_bytes = int(min_limit_gb * (1024 ** 3))

    print(
        f"[MEMORY] Free GPU memory: {free_gb:.2f} GB / {total_gb:.2f} GB "
        f"(minimum required: {min_limit_gb:.2f} GB)"
    )

    if free_bytes < min_bytes:
        msg = (
            f"Free GPU memory ({free_gb:.2f} GB) is below configurable minimum "
            f"({min_limit_gb:.2f} GB). Total memory: {total_gb:.2f} GB."
        )
        print(f"[MEMORY] ERROR: {msg}")
        raise RuntimeError(msg)

    return free_bytes, total_bytes


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


def prepare_for_comfy(
    min_free_gb=None,
    min_free_bytes=None,
):
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

    if torch is not None and torch.cuda.is_available():
        check_free_gpu_memory(
            min_free_gb=min_free_gb,
            min_free_bytes=min_free_bytes,
        )