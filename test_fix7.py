"""Test Fix 7: motion LoRA on both experts (video_generator.py).
Task: add a second LoraLoaderModelOnly for the low-noise path (node 6 -> node 8),
with the same strength, and attach the LoRA to both. If no LoRA name is given,
leave the workflow exactly as it is now.
Test: build the workflow with and without a LoRA and print the node graph;
with LoRA, both ModelSamplingSD3 nodes must read from a motion-LoRA node.
"""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

from app.config import WAN_I2V_WORKFLOW
from app.video_generator import VideoGenerator

print("=" * 60)
print("Test Fix 7: motion LoRA on both Wan experts")
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


# Resolve workflow path locally if default container path does not exist
workflow_path = Path(WAN_I2V_WORKFLOW)
if not workflow_path.exists():
    workflow_path = Path(__file__).resolve().parent / "workflows" / "wan2.2_i2v_api.json"

with open(workflow_path, "r", encoding="utf-8") as f:
    base_workflow = json.load(f)

# Helper to run generate with mock ComfyUI execution
captured_workflows = {}


def run_generator_test(case_name, motion_lora_name=None, motion_lora_strength=0.65):
    vg = VideoGenerator()
    test_input_dir = Path("temp_comfy_input")
    test_input_dir.mkdir(parents=True, exist_ok=True)
    vg.input_dir = test_input_dir

    dummy_keyframe = Path("temp_test_keyframe.png")
    dummy_keyframe.write_bytes(b"dummy_png_bytes")
    dummy_output = Path("temp_test_output.mp4")

    def mock_load_workflow(path):
        return copy.deepcopy(base_workflow)

    def mock_submit(workflow):
        captured_workflows[case_name] = copy.deepcopy(workflow)
        return "mock_prompt_id", "mock_client_id"

    def mock_wait(prompt_id, timeout=1800):
        return {
            "outputs": {
                "16": {
                    "images": [{"filename": "test.mp4"}],
                    "animated": [True],
                }
            }
        }

    vg.runner.load_workflow = mock_load_workflow
    vg.runner.submit = mock_submit
    vg.runner.wait = mock_wait
    vg.download_video = MagicMock()

    try:
        vg.generate(
            keyframe=dummy_keyframe,
            motion_prompt="test motion",
            output_path=dummy_output,
            seed=42,
            motion_lora_name=motion_lora_name,
            motion_lora_strength=motion_lora_strength,
        )
    finally:
        if dummy_keyframe.exists():
            dummy_keyframe.unlink()
        if dummy_output.exists():
            dummy_output.unlink()
        if test_input_dir.exists():
            for f in test_input_dir.iterdir():
                f.unlink()
            test_input_dir.rmdir()

    return captured_workflows[case_name]


# ----------------------------------------------------
# 1. Without motion LoRA (motion_lora_name=None)
# ----------------------------------------------------
print("\n--- 1. Workflow without motion LoRA ---")
wf_without = run_generator_test("without_lora", motion_lora_name=None)

node_7_input = wf_without["7"]["inputs"]["model"]
node_8_input = wf_without["8"]["inputs"]["model"]

print("  Node graph for sampling experts (without LoRA):")
print(f"    High-noise path: Node 5 ({wf_without['5']['class_type']}) -> Node 7 ({wf_without['7']['class_type']}) [reads {node_7_input}]")
print(f"    Low-noise path:  Node 6 ({wf_without['6']['class_type']}) -> Node 8 ({wf_without['8']['class_type']}) [reads {node_8_input}]")

check("Node 500 absent without LoRA", "500" in wf_without, False)
check("Node 501 absent without LoRA", "501" in wf_without, False)
check("Node 7 (high noise) reads from node 5", node_7_input, ["5", 0])
check("Node 8 (low noise) reads from node 6", node_8_input, ["6", 0])

# ----------------------------------------------------
# 2. With motion LoRA (default strength 0.65)
# ----------------------------------------------------
print("\n--- 2. Workflow with motion LoRA (strength 0.65) ---")
test_lora_name = "wan2.2_multi_motion_v1.safetensors"
wf_with = run_generator_test("with_lora", motion_lora_name=test_lora_name, motion_lora_strength=0.65)

node_7_input_with = wf_with["7"]["inputs"]["model"]
node_8_input_with = wf_with["8"]["inputs"]["model"]

print("  Node graph for sampling experts (with LoRA):")
print(f"    High-noise path: Node 5 -> Node 500 ({wf_with.get('500', {}).get('class_type')}) -> Node 7 ({wf_with['7']['class_type']}) [reads {node_7_input_with}]")
print(f"    Low-noise path:  Node 6 -> Node 501 ({wf_with.get('501', {}).get('class_type')}) -> Node 8 ({wf_with['8']['class_type']}) [reads {node_8_input_with}]")

check("Node 500 added for high-noise path", "500" in wf_with, True)
check("Node 500 class_type is LoraLoaderModelOnly", wf_with.get("500", {}).get("class_type"), "LoraLoaderModelOnly")
check("Node 500 inputs model is ['5', 0]", wf_with.get("500", {}).get("inputs", {}).get("model"), ["5", 0])
check("Node 500 lora_name matches", wf_with.get("500", {}).get("inputs", {}).get("lora_name"), test_lora_name)
check("Node 500 strength is 0.65", wf_with.get("500", {}).get("inputs", {}).get("strength_model"), 0.65)

check("Node 501 added for low-noise path", "501" in wf_with, True)
check("Node 501 class_type is LoraLoaderModelOnly", wf_with.get("501", {}).get("class_type"), "LoraLoaderModelOnly")
check("Node 501 inputs model is ['6', 0]", wf_with.get("501", {}).get("inputs", {}).get("model"), ["6", 0])
check("Node 501 lora_name matches", wf_with.get("501", {}).get("inputs", {}).get("lora_name"), test_lora_name)
check("Node 501 strength is 0.65", wf_with.get("501", {}).get("inputs", {}).get("strength_model"), 0.65)

check("Node 7 (high noise ModelSamplingSD3) reads from node 500", node_7_input_with, ["500", 0])
check("Node 8 (low noise ModelSamplingSD3) reads from node 501", node_8_input_with, ["501", 0])

# ----------------------------------------------------
# 3. With motion LoRA (custom strength 0.4)
# ----------------------------------------------------
print("\n--- 3. Workflow with custom LoRA strength 0.4 ---")
wf_custom = run_generator_test("with_custom_strength", motion_lora_name=test_lora_name, motion_lora_strength=0.4)
check("Node 500 has custom strength 0.4", wf_custom.get("500", {}).get("inputs", {}).get("strength_model"), 0.4)
check("Node 501 has custom strength 0.4", wf_custom.get("501", {}).get("inputs", {}).get("strength_model"), 0.4)

print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed")
print("=" * 60)

if failed > 0:
    sys.exit(1)

