#!/usr/bin/env python3
"""
validate_fun_control_workflow.py - Workflow validation script for Wan 2.2 14B Fun Control.

Validates workflows/wan2.2_fun_control_api.json against ComfyUI's live /object_info endpoint:
1. Loads the workflow JSON file.
2. Queries GET <comfy_url>/object_info.
3. For EVERY node in the workflow:
   - Verifies class_type exists in object_info.
   - Verifies every input name used in the workflow exists in the node's schema (required/optional/hidden).
   - Verifies all required schema inputs are provided.
   - For model loaders (UNETLoader, LoraLoaderModelOnly, CLIPLoader, VAELoader), checks that
     the referenced model file name exists in ComfyUI's detected model file list.
4. Checks that all node IDs in FUN_CONTROL_NODE_IDS (app/config.py) exist in the workflow
   and match the expected class_type.
5. Prints PASS/FAIL per node with detailed diffs on failure, and exits non-zero if any check fails.

DOES NOT submit anything to /prompt. ONLY performs read-only inspection via /object_info.

Usage:
  python3 cloning/validate_fun_control_workflow.py [--comfy-url http://127.0.0.1:8188] [--workflow workflows/wan2.2_fun_control_api.json]
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Attempt to load COMFY_URL and FUN_CONTROL_NODE_IDS from app.config
try:
    from app.config import (
        COMFY_URL,
        FUN_CONTROL_NODE_IDS,
        FUN_CONTROL_WORKFLOW,
        WORKFLOWS_DIR,
    )
except ImportError:
    COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188")
    WORKFLOWS_DIR = PROJECT_ROOT / "workflows"
    FUN_CONTROL_WORKFLOW = WORKFLOWS_DIR / "wan2.2_fun_control_api.json"
    FUN_CONTROL_NODE_IDS = {
        "unet_high": "3",
        "unet_low": "4",
        "lora_high": "5",
        "lora_low": "6",
        "shift_high": "7",
        "shift_low": "8",
        "prompt_pos": "9",
        "prompt_neg": "10",
        "fun_control": "11",
        "sampler_high": "12",
        "sampler_low": "13",
        "vae_decode": "14",
        "create_video": "15",
        "save_video": "16",
        "load_control_video": "20",
        "get_video_components": "21",
        "load_ref_image": "1000",
    }

# Mapping of FUN_CONTROL_NODE_IDS role names to expected class types
EXPECTED_ROLE_CLASSES = {
    "unet_high": ["UNETLoader"],
    "unet_low": ["UNETLoader"],
    "lora_high": ["LoraLoaderModelOnly"],
    "lora_low": ["LoraLoaderModelOnly"],
    "shift_high": ["ModelSamplingSD3", "ModelSamplingWan"],
    "shift_low": ["ModelSamplingSD3", "ModelSamplingWan"],
    "prompt_pos": ["CLIPTextEncode"],
    "prompt_neg": ["CLIPTextEncode"],
    "fun_control": ["Wan22FunControlToVideo"],
    "sampler_high": ["KSamplerAdvanced"],
    "sampler_low": ["KSamplerAdvanced"],
    "vae_decode": ["VAEDecode"],
    "create_video": ["CreateVideo"],
    "save_video": ["SaveVideo"],
    "load_control_video": ["LoadVideo"],
    "get_video_components": ["GetVideoComponents"],
    "load_ref_image": ["LoadImage"],
}

# Loader nodes and their corresponding model file parameter names
MODEL_LOADER_PARAMS = {
    "UNETLoader": "unet_name",
    "LoraLoaderModelOnly": "lora_name",
    "CLIPLoader": "clip_name",
    "VAELoader": "vae_name",
}


def fetch_object_info(comfy_url: str, timeout: int = 15) -> Dict[str, Any]:
    """Queries ComfyUI GET /object_info endpoint."""
    clean_url = comfy_url.rstrip("/")
    target = f"{clean_url}/object_info"
    req = urllib.request.Request(
        target,
        headers={"User-Agent": "WanFunControlValidator/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} from {target}")
        data = response.read()
        return json.loads(data.decode("utf-8"))


def validate_node(
    node_id: str,
    node_data: Dict[str, Any],
    object_info: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Validates a single workflow node against ComfyUI schema in object_info:
    - class_type exists.
    - inputs exist in schema.
    - required schema inputs are satisfied.
    - model loader filenames exist in ComfyUI's model list.
    """
    errors: List[str] = []
    class_type = node_data.get("class_type")

    if not class_type:
        errors.append(f"Node '{node_id}' is missing 'class_type'")
        return False, errors

    if class_type not in object_info:
        errors.append(
            f"Class '{class_type}' not found in ComfyUI object_info (custom node pack missing or uninstalled)"
        )
        return False, errors

    schema = object_info[class_type]
    input_schema = schema.get("input", {})
    required_schema: Dict[str, Any] = input_schema.get("required", {})
    optional_schema: Dict[str, Any] = input_schema.get("optional", {})
    hidden_schema: Dict[str, Any] = input_schema.get("hidden", {})

    valid_keys = set(required_schema.keys()) | set(optional_schema.keys()) | set(hidden_schema.keys())

    provided_inputs: Dict[str, Any] = node_data.get("inputs", {})

    # 1. Check all provided inputs exist in node's schema
    for inp_name in provided_inputs:
        if inp_name not in valid_keys:
            errors.append(
                f"Unknown input parameter '{inp_name}' for class '{class_type}'. "
                f"Valid inputs: {sorted(list(valid_keys))}"
            )

    # 2. Check all required schema parameters are provided
    for req_name, req_spec in required_schema.items():
        if req_name not in provided_inputs:
            errors.append(
                f"Missing required input parameter '{req_name}' (expected type: {req_spec[0] if req_spec else 'unknown'})"
            )

    # 3. For model loaders, verify model filename appears in ComfyUI's detected files list
    if class_type in MODEL_LOADER_PARAMS:
        param_name = MODEL_LOADER_PARAMS[class_type]
        if param_name in provided_inputs:
            chosen_file = provided_inputs[param_name]
            # In ComfyUI object_info, the available files are passed as a list in required_schema[param_name][0]
            if param_name in required_schema and isinstance(required_schema[param_name], list) and required_schema[param_name]:
                avail_files = required_schema[param_name][0]
                if isinstance(avail_files, list):
                    if chosen_file not in avail_files:
                        errors.append(
                            f"Model file '{chosen_file}' for '{param_name}' is not in ComfyUI's model list. "
                            f"Available files ({len(avail_files)}): {avail_files[:5]}..."
                        )

    is_pass = len(errors) == 0
    return is_pass, errors


def validate_config_node_ids(
    workflow: Dict[str, Any],
    node_ids_config: Dict[str, str],
) -> Tuple[bool, List[str]]:
    """
    Checks that every role in FUN_CONTROL_NODE_IDS:
    - Exists in the workflow JSON.
    - Has an expected class_type.
    """
    errors: List[str] = []
    for role, expected_id in node_ids_config.items():
        if str(expected_id) not in workflow:
            errors.append(
                f"Configured node ID for '{role}' (ID: '{expected_id}') does not exist in workflow JSON"
            )
            continue

        actual_node = workflow[str(expected_id)]
        actual_class = actual_node.get("class_type")

        expected_classes = EXPECTED_ROLE_CLASSES.get(role)
        if expected_classes and actual_class not in expected_classes:
            errors.append(
                f"Node ID '{expected_id}' for '{role}' has class_type '{actual_class}', "
                f"expected one of {expected_classes}"
            )

    is_pass = len(errors) == 0
    return is_pass, errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate workflows/wan2.2_fun_control_api.json against ComfyUI's live /object_info endpoint"
    )
    parser.add_argument(
        "--comfy-url",
        default=COMFY_URL,
        help=f"ComfyUI base URL (default: {COMFY_URL})",
    )
    parser.add_argument(
        "--workflow",
        default=str(FUN_CONTROL_WORKFLOW),
        help=f"Path to workflow API JSON (default: {FUN_CONTROL_WORKFLOW})",
    )
    parser.add_argument(
        "--mock-object-info",
        default=None,
        help="Optional path to mock object_info.json for offline testing",
    )
    args = parser.parse_args()

    workflow_path = Path(args.workflow)
    if not workflow_path.exists():
        print(f"FAIL: Workflow file does not exist: {workflow_path}")
        sys.exit(1)

    print("=" * 75)
    print("  WAN 2.2 FUN CONTROL WORKFLOW VALIDATOR")
    print("=" * 75)
    print(f"Workflow File: {workflow_path.resolve()}")
    print(f"ComfyUI URL  : {args.comfy_url}")
    print("=" * 75)

    # 1. Load workflow JSON
    try:
        with open(workflow_path, "r", encoding="utf-8") as wf_file:
            workflow: Dict[str, Any] = json.load(wf_file)
    except Exception as e:
        print(f"FAIL: Could not parse workflow JSON file: {e}")
        sys.exit(1)

    # 2. Fetch ComfyUI /object_info
    if args.mock_object_info:
        print(f"Loading mock object_info from: {args.mock_object_info}")
        with open(args.mock_object_info, "r", encoding="utf-8") as mf:
            object_info = json.load(mf)
    else:
        print(f"Querying GET {args.comfy_url.rstrip('/')}/object_info (no prompts submitted)...")
        try:
            object_info = fetch_object_info(args.comfy_url)
            print(f"Successfully retrieved object_info ({len(object_info)} registered node types).\n")
        except urllib.error.URLError as e:
            print(f"\n[FAIL] Could not connect to ComfyUI at {args.comfy_url}: {e}")
            print("Ensure ComfyUI is running on the specified URL or pass --comfy-url <URL>.")
            sys.exit(1)
        except Exception as e:
            print(f"\n[FAIL] Error fetching object_info: {e}")
            sys.exit(1)

    # 3. Validate every node in workflow
    total_nodes = len(workflow)
    pass_count = 0
    fail_count = 0
    node_failures: Dict[str, List[str]] = {}

    # Sort node keys numerically for clean output
    def node_sort_key(k: str):
        try:
            return (0, int(k))
        except ValueError:
            return (1, k)

    sorted_node_ids = sorted(workflow.keys(), key=node_sort_key)

    print("--- NODE-BY-NODE SCHEMA VALIDATION ---")
    for nid in sorted_node_ids:
        ndata = workflow[nid]
        passed, errs = validate_node(nid, ndata, object_info)
        class_type = ndata.get("class_type", "Unknown")
        if passed:
            pass_count += 1
            print(f"  [PASS] Node {nid:>4} ({class_type})")
        else:
            fail_count += 1
            node_failures[nid] = errs
            print(f"  [FAIL] Node {nid:>4} ({class_type}):")
            for err in errs:
                print(f"         ! {err}")

    # 4. Check FUN_CONTROL_NODE_IDS consistency
    print("\n--- FUN_CONTROL_NODE_IDS CONFIGURATION CHECK ---")
    cfg_passed, cfg_errs = validate_config_node_ids(workflow, FUN_CONTROL_NODE_IDS)
    if cfg_passed:
        print("  [PASS] All node IDs in FUN_CONTROL_NODE_IDS exist and match expected class types.")
    else:
        print("  [FAIL] Mismatches found in FUN_CONTROL_NODE_IDS:")
        for cerr in cfg_errs:
            print(f"         ! {cerr}")

    # 5. Summary and exit
    print("\n" + "=" * 75)
    print("  VALIDATION SUMMARY")
    print("=" * 75)
    print(f"  Total Workflow Nodes : {total_nodes}")
    print(f"  Passed Nodes         : {pass_count}")
    print(f"  Failed Nodes         : {fail_count}")
    print(f"  Config Check Passed  : {cfg_passed}")
    print("=" * 75)

    if fail_count > 0 or not cfg_passed:
        print("\nRESULT: OVERALL VALIDATION FAILED. Review errors above.")
        sys.exit(1)
    else:
        print("\nRESULT: OVERALL VALIDATION PASSED! Workflow is 100% compliant with ComfyUI schema.")
        sys.exit(0)


if __name__ == "__main__":
    main()
