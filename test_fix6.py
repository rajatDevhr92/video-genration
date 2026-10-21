"""Test Fix 6: free GPU memory (gpu_memory.py, local_llm_provider.py).
Task: add an unload() to LocalLLMProvider that deletes the model and tokenizer,
runs gc.collect() and torch.cuda.empty_cache(), then checks free GPU memory with
torch.cuda.mem_get_info(). Call it after planning and before the first ComfyUI call.
Log the free memory before and after. If free memory is below a configurable minimum,
raise a clear error.
Test: log shows memory freed after unload.
"""
import io
import os
import sys
from unittest.mock import MagicMock, patch

from app.gpu_memory import (
    check_free_gpu_memory,
    get_free_gpu_memory,
    prepare_for_comfy,
)
from app.local_llm_provider import LocalLLMProvider

print("=" * 60)
print("Test Fix 6: free GPU memory")
print("=" * 60)

passed = 0
failed = 0


def check(label, result, expected):
    global passed, failed
    ok = result == expected
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: got={result}, expected={expected}")
    if ok:
        passed += 1
    else:
        failed += 1


# Helper to create a dummy provider without calling __init__
def create_dummy_provider():
    provider = LocalLLMProvider.__new__(LocalLLMProvider)
    provider.model = MagicMock(name="FakeModel")
    provider.tokenizer = MagicMock(name="FakeTokenizer")
    provider.max_new_tokens = 1024
    provider.model_path = "mock-model"
    return provider


# ----------------------------------------------------
# 1. Unload deletes model and tokenizer
# ----------------------------------------------------
print("\n1. Unload deletes model and tokenizer attributes")
p = create_dummy_provider()
check("model exists before unload", hasattr(p, "model") and p.model is not None, True)
check("tokenizer exists before unload", hasattr(p, "tokenizer") and p.tokenizer is not None, True)

p.unload()

check("model is None after unload", getattr(p, "model", None) is None, True)
check("tokenizer is None after unload", getattr(p, "tokenizer", None) is None, True)


# ----------------------------------------------------
# 2. Free memory logged before and after unload
# ----------------------------------------------------
print("\n2. Free memory logged before and after unload with torch.cuda.mem_get_info")
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = True

# Simulate memory increasing after model is unloaded:
# Before: 2 GB free (2 * 1024^3), After: 14 GB free (14 * 1024^3) out of 16 GB total
mem_readings = [
    (2 * 1024**3, 16 * 1024**3),   # before unload
    (14 * 1024**3, 16 * 1024**3),  # after unload
]
mock_torch.cuda.mem_get_info.side_effect = mem_readings

p2 = create_dummy_provider()

captured_output = io.StringIO()
old_stdout = sys.stdout
sys.stdout = captured_output

with patch("app.local_llm_provider.torch", mock_torch), \
     patch("app.gpu_memory.torch", mock_torch):
    p2.unload(min_free_gb=10.0)

sys.stdout = old_stdout
logs = captured_output.getvalue()

check("logged free memory before unload", "Free GPU memory before unload: 2.00 GB / 16.00 GB" in logs, True)
check("logged free memory after unload", "Free GPU memory after unload: 14.00 GB / 16.00 GB" in logs, True)
check("torch.cuda.empty_cache called", mock_torch.cuda.empty_cache.called, True)


# ----------------------------------------------------
# 3. Configurable minimum check (raises when below minimum)
# ----------------------------------------------------
print("\n3. Low memory check raises RuntimeError")

mock_torch_low = MagicMock()
mock_torch_low.cuda.is_available.return_value = True
# Only 3 GB free after unload
mock_torch_low.cuda.mem_get_info.side_effect = [
    (1 * 1024**3, 16 * 1024**3),  # before
    (3 * 1024**3, 16 * 1024**3),  # after
]

p3 = create_dummy_provider()
raised_min_error = False
try:
    with patch("app.local_llm_provider.torch", mock_torch_low), \
         patch("app.gpu_memory.torch", mock_torch_low):
        p3.unload(min_free_gb=8.0)
except RuntimeError as e:
    raised_min_error = True
    check("error mentions below configurable minimum", "below configurable minimum" in str(e), True)

check("raised RuntimeError when free memory below min_free_gb", raised_min_error, True)


# ----------------------------------------------------
# 4. Configurable via MIN_FREE_GPU_MEMORY_GB environment variable
# ----------------------------------------------------
print("\n4. Minimum configurable via environment variable MIN_FREE_GPU_MEMORY_GB")

mock_torch_env = MagicMock()
mock_torch_env.cuda.is_available.return_value = True
mock_torch_env.cuda.mem_get_info.side_effect = [
    (2 * 1024**3, 16 * 1024**3),
    (5 * 1024**3, 16 * 1024**3),
]

p4 = create_dummy_provider()
env_raised = False
os.environ["MIN_FREE_GPU_MEMORY_GB"] = "10.0"

try:
    with patch("app.local_llm_provider.torch", mock_torch_env), \
         patch("app.gpu_memory.torch", mock_torch_env):
        p4.unload()
except RuntimeError as e:
    env_raised = True
    check("env var MIN_FREE_GPU_MEMORY_GB triggered error", "10.00 GB" in str(e), True)
finally:
    del os.environ["MIN_FREE_GPU_MEMORY_GB"]

check("raised RuntimeError when below env var minimum", env_raised, True)


# ----------------------------------------------------
# 5. check_free_gpu_memory & prepare_for_comfy in gpu_memory.py
# ----------------------------------------------------
print("\n5. gpu_memory helpers: check_free_gpu_memory and prepare_for_comfy")

mock_torch_comfy = MagicMock()
mock_torch_comfy.cuda.is_available.return_value = True
mock_torch_comfy.cuda.mem_get_info.return_value = (12 * 1024**3, 16 * 1024**3)

with patch("app.gpu_memory.torch", mock_torch_comfy):
    free_b, total_b = get_free_gpu_memory()
    check("get_free_gpu_memory returns free bytes", free_b, 12 * 1024**3)
    check("get_free_gpu_memory returns total bytes", total_b, 16 * 1024**3)

    # Above minimum passes
    free_chk, total_chk = check_free_gpu_memory(min_free_gb=10.0)
    check("check_free_gpu_memory passes when above min", free_chk, 12 * 1024**3)

    # prepare_for_comfy runs and checks memory
    prepare_for_comfy(min_free_gb=5.0)
    check("prepare_for_comfy executed cleanup and check", mock_torch_comfy.cuda.empty_cache.called, True)


# ----------------------------------------------------
# 6. Low memory in prepare_for_comfy raises RuntimeError
# ----------------------------------------------------
print("\n6. prepare_for_comfy raises on low memory")

mock_torch_fail = MagicMock()
mock_torch_fail.cuda.is_available.return_value = True
mock_torch_fail.cuda.mem_get_info.return_value = (1 * 1024**3, 16 * 1024**3)

comfy_failed = False
try:
    with patch("app.gpu_memory.torch", mock_torch_fail):
        prepare_for_comfy(min_free_gb=5.0)
except RuntimeError as e:
    comfy_failed = True
    check("prepare_for_comfy error message has details", "below configurable minimum" in str(e), True)

check("prepare_for_comfy raises RuntimeError when free memory is low", comfy_failed, True)


print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed")
print("=" * 60)

if failed > 0:
    sys.exit(1)

