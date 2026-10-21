"""Test Fix 3: remove anatomy keyword fallback in continuity.py."""
import sys
import os

from app.continuity import CharState, needs_anatomy_pass, ANATOMY_ACTS, ACTS

print("=" * 60)
print("Test Fix 3: anatomy-pass fallback removal")
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


# Setup test states
char_dressed = CharState(character_id="c1", name="Alice", gender="female", outfit="dress", top_on=True, bottom_on=True)
char_topless = CharState(character_id="c2", name="Bob", gender="male", outfit="suit", top_on=False, bottom_on=True)
char_unclothed = CharState(character_id="c3", name="Charlie", gender="male", outfit="jeans", top_on=False, bottom_on=False)
char_bottomless = CharState(character_id="c4", name="Diana", gender="female", outfit="skirt", top_on=True, bottom_on=False)

states_dressed = {"c1": char_dressed, "c2": char_topless}
states_unclothed = {"c1": char_dressed, "c3": char_unclothed}
states_bottomless = {"c4": char_bottomless}

# Test 1: Primary test from Untitled.md:
# "a scene whose free text contains a fragment but whose act is not in the closed list must return False"
print("\n1. Free text fragment with non-anatomy act (must return False even when unclothed)")
scene_with_fragments = {
    "description": "deep penetrating conversation while touching passionately",
    "action": "oral presentation near the bookshelf with vigorous thrusting motions",
    "penetration_state": "penetrating gaze",
}

for act in ["kiss", "touch", "none", "other_contact"]:
    res = needs_anatomy_pass(act, states_unclothed, scene=scene_with_fragments)
    check(f"act='{act}' with fragment-laden scene text and unclothed chars -> False", res, False)

# Test 2: Anatomy act with unclothed characters (must return True)
print("\n2. Closed anatomy acts with unclothed characters (must return True)")
for act in ["oral", "vaginal", "anal", "ORAL", " Vaginal "]:
    res = needs_anatomy_pass(act, states_unclothed, scene=scene_with_fragments)
    check(f"act='{act}' with unclothed character -> True", res, True)

# Test 3: Anatomy act with bottomless character (must return True)
print("\n3. Closed anatomy acts with bottomless character (must return True)")
res = needs_anatomy_pass("vaginal", states_bottomless)
check("act='vaginal' with bottomless character -> True", res, True)

# Test 4: Anatomy act with dressed characters (must return False)
print("\n4. Closed anatomy acts with fully dressed/topless characters (must return False)")
for act in ["oral", "vaginal", "anal"]:
    res = needs_anatomy_pass(act, states_dressed, scene=scene_with_fragments)
    check(f"act='{act}' with dressed characters -> False", res, False)

# Test 5: Empty / None / invalid act (must return False)
print("\n5. Invalid / None act (must return False)")
check("act=None -> False", needs_anatomy_pass(None, states_unclothed), False)
check("act='' -> False", needs_anatomy_pass("", states_unclothed), False)
check("act='unclear' -> False", needs_anatomy_pass("unclear", states_unclothed), False)

# Test 6: Verify ANATOMY_ACTS is a subset of ACTS
print("\n6. ANATOMY_ACTS consistency")
check("ANATOMY_ACTS is subset of ACTS", ANATOMY_ACTS.issubset(ACTS), True)
check("ANATOMY_ACTS exact members", ANATOMY_ACTS, {"oral", "vaginal", "anal"})

print(f"\n{'=' * 60}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'=' * 60}")
sys.exit(1 if failed else 0)

