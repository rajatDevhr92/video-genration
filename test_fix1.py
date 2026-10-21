"""Test Fix 1: audio gender whole-word matching."""
import sys, os
import importlib.util, types

# Load just the module without triggering OpenAI init
spec = importlib.util.spec_from_file_location(
    "audio_generator",
    os.path.join(os.path.dirname(__file__), "app", "audio_generator.py"),
)
mod = importlib.util.module_from_spec(spec)

# Patch dotenv and openai so they don't fail
sys.modules["dotenv"] = types.ModuleType("dotenv")
sys.modules["dotenv"].load_dotenv = lambda *a, **kw: None
sys.modules["openai"] = None  # triggers the ImportError path

spec.loader.exec_module(mod)
resolve_character_voice = mod.resolve_character_voice

import logging
logging.basicConfig(level=logging.DEBUG)

print("=" * 60)
print("Test Fix 1: audio gender whole-word matching")
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

# Test 1: "the shelf is there" must NOT pick a female voice from gender detection.
# Use char_099 to avoid the character-id fallback (001->nova, 002->onyx).
print("\n1. Substring false-positive test")
char_shelf = {"appearance": "the shelf is there", "name": "Bob"}
v = resolve_character_voice("char_099", char_shelf)
# No gender detected -> char_099 falls through to "echo" (not nova/onyx)
check('"the shelf is there" (char_099) -> should NOT be female (nova)', v != "nova", True)

# Test 1b: same with char_001 — gender must be None (falls to id heuristic)
char_shelf2 = {"appearance": "the shelf is there", "name": "Bob"}
v2 = resolve_character_voice("char_001", char_shelf2)
# char_001 id-fallback returns "nova", but that's the id heuristic, NOT gender matching.
# The key point: the WARNING log shows no gender was detected from text.
# We test that explicitly:
# gender should be None => the function reaches the character-id fallback.
# Since char_001 maps to nova via the id fallback, that is expected.
check('"the shelf is there" (char_001) -> id fallback gives nova (not gender)', v2, "nova")

# Test 2: "she smiled" must pick a female voice
print("\n2. Female keyword test")
char_she = {"appearance": "she smiled warmly", "name": "Elena"}
v = resolve_character_voice("char_099", char_she)
check('"she smiled" -> should be female (nova)', v, "nova")

# Test 3: Explicit gender from plan takes precedence
print("\n3. Explicit gender from plan")
char_explicit_f = {"gender": "female", "name": "Sarah"}
v = resolve_character_voice("char_099", char_explicit_f)
check('explicit gender=female -> nova', v, "nova")

char_explicit_m = {"gender": "male", "name": "Tom"}
v = resolve_character_voice("char_099", char_explicit_m)
check('explicit gender=male -> onyx', v, "onyx")

# Test 4: "there" and "other" should not trigger female
print("\n4. More false-positive tests")
char_there = {"appearance": "standing there by the other door", "name": "Alex"}
v = resolve_character_voice("char_099", char_there)
check('"there" and "other" -> should NOT be female', v != "nova", True)

# Test 5: "he walked" should pick male
print("\n5. Male keyword test")
char_he = {"appearance": "he walked slowly", "name": "Marcus"}
v = resolve_character_voice("char_099", char_he)
check('"he walked" -> should be male (onyx)', v, "onyx")

# Test 6: "the hero" should NOT match "her" or "he"
print("\n6. 'hero' should not match 'her' or 'he'")
char_hero = {"appearance": "the hero stands tall", "name": "Hero"}
v = resolve_character_voice("char_099", char_hero)
check('"the hero" -> should NOT be female or male from substring', v != "nova" and v != "onyx", True)

# Test 7: Explicit gender overrides conflicting text
print("\n7. Explicit gender overrides text")
char_conflict = {"gender": "male", "appearance": "she looked beautiful", "name": "Chris"}
v = resolve_character_voice("char_099", char_conflict)
check('gender=male but text says "she" -> should be male (onyx)', v, "onyx")

# Test 8: "male" should not match inside "female"
print("\n8. 'female' should not match 'male'")
char_female_text = {"appearance": "a female warrior", "name": "Diana"}
v = resolve_character_voice("char_099", char_female_text)
check('"a female warrior" -> should be female (nova), not male', v, "nova")

print(f"\n{'=' * 60}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'=' * 60}")
sys.exit(1 if failed else 0)

