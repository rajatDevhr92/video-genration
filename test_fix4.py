"""Test Fix 4: planner fallbacks, strict mode, and fallback logging."""
import os
import sys
import shutil
import tempfile
from pathlib import Path

from app.story_planner import StoryPlanner, PlanError, validate_plan

print("=" * 60)
print("Test Fix 4: planner fallbacks & strict mode")
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

def make_valid_plan():
    return {
        "title": "A Day at the Lake",
        "summary": "Two friends relax by the water.",
        "characters": [
            {
                "id": "char_001",
                "name": "Sarah",
                "gender": "female",
                "appearance": "Adult woman, 28 years old, athletic build",
                "clothing": "Summer dress",
                "personality": "Cheerful and energetic",
                "visual_anchor": "Adult woman, athletic build",
            }
        ],
        "locations": [
            {
                "id": "loc_001",
                "name": "Lake Cabin",
                "description": "A wooden cabin on the lakeside",
                "visual_style": "Warm natural sunlight",
            }
        ],
        "scenes": [
            {
                "id": "scene_001",
                "order": 1,
                "duration": 5.0,
                "location_id": "loc_001",
                "character_ids": ["char_001"],
                "act": "none",
                "wardrobe_events": [],
                "clothing_state": "Canonical outfit from character reference",
                "pose_type": "Standing upright",
                "primary_motion": "Sarah walks toward the shoreline",
                "motion_intensity": "medium",
                "expression": "Warm smile",
                "description": "Sarah walks toward the shoreline under the sun.",
                "action": "Sarah walks toward the water.",
                "camera": "Medium shot, side angle (matches CENTER PANEL 3/4 Side View of turnaround sheet)",
                "lighting": "Warm morning sunlight",
                "video_prompt": "Sarah walks toward water. Exact contact: None. Hands: resting naturally. Bodies remain locked in Standing upright. Camera: Medium shot.",
                "audio": {
                    "narration": [{"text": "The lake was peaceful in the morning.", "voice": "alloy"}],
                    "dialogue": [],
                    "sound_effects": ["gentle water lapping"],
                    "music": {"mood": "peaceful", "intensity": "low"},
                },
            }
        ],
    }

tmp_dir = Path(tempfile.mkdtemp(prefix="test_fix4_"))
try:
    # ----------------------------------------------------
    # Test 1: Valid plan passes strict mode with 0 fallbacks
    # ----------------------------------------------------
    print("\n1. Valid plan in strict mode")
    p1 = make_valid_plan()
    planner_strict = StoryPlanner(provider=None, strict=True, job_dir=tmp_dir)
    validated1 = planner_strict.validate(p1)
    check("valid plan passes in strict mode", validated1 is not None, True)
    check("fallback total is 0", validated1["meta"]["fallbacks"]["total"], 0)
    check("fallback counts equal 0", validated1["meta"]["fallbacks"] == 0, True)

    # ----------------------------------------------------
    # Test 2: Missing field in strict mode raises PlanError
    # ----------------------------------------------------
    print("\n2. Missing field in strict mode (must raise PlanError)")
    p2 = make_valid_plan()
    del p2["scenes"][0]["lighting"]  # remove required field

    raised = False
    try:
        planner_strict.validate(p2)
    except PlanError as e:
        raised = True
        print(f"     Expected PlanError raised: {e}")
    check("strict mode raises PlanError on missing field", raised, True)

    # ----------------------------------------------------
    # Test 3: Missing field in non-strict mode logs ONE line
    # ----------------------------------------------------
    print("\n3. Missing field in non-strict mode (must log ONE line to fallbacks.log)")
    job_folder = tmp_dir / "job_001"
    job_folder.mkdir(parents=True, exist_ok=True)
    fallback_log = job_folder / "fallbacks.log"

    p3 = make_valid_plan()
    del p3["scenes"][0]["lighting"]  # remove required field

    planner_non_strict = StoryPlanner(provider=None, strict=False, job_dir=job_folder)
    validated3 = planner_non_strict.validate(p3)

    check("non-strict mode does not raise", validated3 is not None, True)
    check("lighting was populated with fallback", "lighting" in validated3["scenes"][0], True)
    check("fallback count total is 1", validated3["meta"]["fallbacks"]["total"], 1)
    check("fallback count by_field has lighting", validated3["meta"]["fallbacks"]["lighting"], 1)
    check("fallback count equals int 1", validated3["meta"]["fallbacks"] == 1, True)
    check("fallbacks.log file exists", fallback_log.exists(), True)

    log_lines = fallback_log.read_text(encoding="utf-8").strip().splitlines()
    check("fallbacks.log contains exactly 1 line", len(log_lines), 1)
    if log_lines:
        print(f"     Logged line: {log_lines[0]}")
        check("log line mentions scene number", "Scene 1" in log_lines[0], True)
        check("log line mentions field name", "lighting" in log_lines[0], True)

    # ----------------------------------------------------
    # Test 4: Default strict=True on StoryPlanner and validate_plan
    # ----------------------------------------------------
    print("\n4. Default strict behavior")
    p4 = make_valid_plan()
    del p4["scenes"][0]["camera"]

    planner_default = StoryPlanner(provider=None)
    check("StoryPlanner default strict is True", planner_default.strict, True)

    raised_default = False
    try:
        planner_default.validate(p4)
    except PlanError:
        raised_default = True
    check("default planner raises PlanError on missing camera", raised_default, True)

    raised_helper = False
    try:
        validate_plan(p4)
    except PlanError:
        raised_helper = True
    check("validate_plan helper raises PlanError by default", raised_helper, True)

    # ----------------------------------------------------
    # Test 5: Multiple missing fields in non-strict mode
    # ----------------------------------------------------
    print("\n5. Multiple missing fields in non-strict mode")
    job_folder_multi = tmp_dir / "job_multi"
    job_folder_multi.mkdir(parents=True, exist_ok=True)
    multi_log = job_folder_multi / "fallbacks.log"

    p5 = make_valid_plan()
    del p5["scenes"][0]["lighting"]
    del p5["scenes"][0]["camera"]

    planner_multi = StoryPlanner(provider=None, strict=False, job_dir=job_folder_multi)
    validated5 = planner_multi.validate(p5)

    check("multiple missing fields: total is 2", validated5["meta"]["fallbacks"]["total"], 2)
    check("multiple missing fields: lighting count 1", validated5["meta"]["fallbacks"]["lighting"], 1)
    check("multiple missing fields: camera count 1", validated5["meta"]["fallbacks"]["camera"], 1)
    multi_lines = multi_log.read_text(encoding="utf-8").strip().splitlines()
    check("multi fallbacks.log has 2 lines", len(multi_lines), 2)

finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)

print(f"\n{'=' * 60}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'=' * 60}")
sys.exit(1 if failed else 0)

