"""Test Fix 5: adult-age rule (story_planner.py).
Test: a cast with age 17, a missing age, and age "young" must each be rejected; age 25 must pass.
"""
import copy
from app.story_planner import StoryPlanner, PlanError, validate_plan

print("=" * 60)
print("Test Fix 5: adult-age rule")
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

def make_base_plan():
    return {
        "title": "A Day at the Lake",
        "summary": "Two friends relax by the water.",
        "characters": [
            {
                "id": "char_001",
                "name": "Sarah",
                "gender": "female",
                "appearance": "Adult woman, athletic build",
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

# ----------------------------------------------------
# 1. Cast with age 25 must pass
# ----------------------------------------------------
print("\n1. Cast with age 25 (must pass)")
p_valid = make_base_plan()
p_valid["characters"][0]["age"] = 25
try:
    res = validate_plan(p_valid)
    check("age 25 passes validation", res is not None, True)
    check("age remains 25", res["characters"][0]["age"], 25)
except Exception as e:
    check(f"age 25 passed without exception, got: {e}", False, True)

# ----------------------------------------------------
# 2. Cast with age 17 must be rejected (underage)
# ----------------------------------------------------
print("\n2. Cast with age 17 (must be rejected)")
p_17 = make_base_plan()
p_17["characters"][0]["age"] = 17
try:
    validate_plan(p_17)
    check("age 17 raised PlanError", False, True)
except PlanError as e:
    err_str = str(e)
    check("age 17 raised PlanError", True, True)
    check("error mentions below 18 / 17", "17" in err_str or "below 18" in err_str, True)
    check("error names the character", "Sarah" in err_str or "char_001" in err_str, True)
except Exception as e:
    check(f"expected PlanError, got {type(e)}", False, True)

# ----------------------------------------------------
# 3. Cast with missing age must be rejected
# ----------------------------------------------------
print("\n3. Cast with missing age (must be rejected)")
p_missing = make_base_plan()
# no "age" key
try:
    validate_plan(p_missing)
    check("missing age raised PlanError", False, True)
except PlanError as e:
    err_str = str(e)
    check("missing age raised PlanError", True, True)
    check("error mentions missing age", "missing" in err_str.lower(), True)
    check("error names the character", "Sarah" in err_str or "char_001" in err_str, True)
except Exception as e:
    check(f"expected PlanError, got {type(e)}", False, True)

# ----------------------------------------------------
# 4. Cast with age "young" must be rejected (unclear)
# ----------------------------------------------------
print("\n4. Cast with age 'young' (must be rejected)")
p_young = make_base_plan()
p_young["characters"][0]["age"] = "young"
try:
    validate_plan(p_young)
    check("age 'young' raised PlanError", False, True)
except PlanError as e:
    err_str = str(e)
    check("age 'young' raised PlanError", True, True)
    check("error mentions unclear / 'young'", "young" in err_str or "unclear" in err_str, True)
    check("error names the character", "Sarah" in err_str or "char_001" in err_str, True)
except Exception as e:
    check(f"expected PlanError, got {type(e)}", False, True)

# ----------------------------------------------------
# 5. Non-strict mode must still reject missing/underage age (no defaulting)
# ----------------------------------------------------
print("\n5. Non-strict mode does not guess or default age")
planner_non_strict = StoryPlanner(provider=None, strict=False)
try:
    planner_non_strict.validate(make_base_plan())
    check("non-strict mode raised PlanError on missing age", False, True)
except PlanError as e:
    check("non-strict mode raises PlanError on missing age", True, True)

p_non_strict_17 = make_base_plan()
p_non_strict_17["characters"][0]["age"] = 17
try:
    planner_non_strict.validate(p_non_strict_17)
    check("non-strict mode raised PlanError on age 17", False, True)
except PlanError as e:
    check("non-strict mode raises PlanError on age 17", True, True)

# ----------------------------------------------------
# 6. Other unclear ages: boolean, float
# ----------------------------------------------------
print("\n6. Boolean and float ages are rejected as unclear")
p_bool = make_base_plan()
p_bool["characters"][0]["age"] = True
try:
    validate_plan(p_bool)
    check("boolean age raised PlanError", False, True)
except PlanError as e:
    check("boolean age raised PlanError", True, True)

p_float = make_base_plan()
p_float["characters"][0]["age"] = 25.5
try:
    validate_plan(p_float)
    check("float age raised PlanError", False, True)
except PlanError as e:
    check("float age raised PlanError", True, True)

# ----------------------------------------------------
# 7. Multiple characters: fails on the first underage character naming them
# ----------------------------------------------------
print("\n7. Multiple characters with one underage character")
p_multi = make_base_plan()
p_multi["characters"][0]["age"] = 30
p_multi["characters"].append({
    "id": "char_002",
    "name": "Tom",
    "gender": "male",
    "age": 16,
    "appearance": "Young guy",
    "clothing": "Jeans and T-shirt",
    "personality": "Quiet",
    "visual_anchor": "Young guy",
})
p_multi["scenes"][0]["character_ids"].append("char_002")
try:
    validate_plan(p_multi)
    check("multi-character with underage character raised PlanError", False, True)
except PlanError as e:
    err_str = str(e)
    check("multi-character with underage character raised PlanError", True, True)
    check("error specifically names Tom / char_002", "Tom" in err_str or "char_002" in err_str, True)

print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed")
print("=" * 60)

if failed > 0:
    exit(1)

