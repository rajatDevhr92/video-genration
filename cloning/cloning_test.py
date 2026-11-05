"""
Unit test for updated cloning schemas, auto-prompting, and safety checks.
Run: python -m cloning.cloning_test
"""
from cloning.cloning_record import CloningRecord
from cloning.cloning_safety import review_media
from cloning.cloning_routes import (
    CharacterTarget,
    CloningJobRequest,
    auto_generate_prompt_for_character,
)

results = {"passed": 0, "failed": 0}


def check(name, actual, expected):
    if actual == expected:
        results["passed"] += 1
        print(f"PASS  {name}")
    else:
        results["failed"] += 1
        print(f"FAIL  {name}: got {actual!r}, expected {expected!r}")


# 1. Test auto prompt generation
prompt_female = auto_generate_prompt_for_character("female", 26)
check("female auto-prompt contains woman", "woman" in prompt_female, True)
check("female auto-prompt contains age", "26" in prompt_female, True)

prompt_male = auto_generate_prompt_for_character("male", 30)
check("male auto-prompt contains man", "man" in prompt_male, True)
check("male auto-prompt contains age", "30" in prompt_male, True)

# 2. Test CharacterTarget Schema
char_without_prompt = CharacterTarget(gender="female", age=25)
check("character without prompt defaults to None", char_without_prompt.character_prompt, None)

char_with_prompt = CharacterTarget(gender="female", age=28, character_prompt="Nordic blonde woman")
check("custom character prompt preserved", char_with_prompt.character_prompt, "Nordic blonde woman")

# 3. Test CloningJobRequest schema flexibility
req_minimal = CloningJobRequest(source_video="test.mp4")
check("request without characters is allowed", req_minimal.characters, None)

req_with_chars = CloningJobRequest(
    source_video="test.mp4",
    characters=[
        CharacterTarget(gender="female", age=24),
        CharacterTarget(gender="male", age=29, character_prompt="Man with glasses"),
    ],
)
check("request with multiple characters accepted", len(req_with_chars.characters), 2)

# 4. Test safety & age checks with mocked face detector
def sampler(video, output_dir):
    return ["frame_1.png", "frame_2.png"]


def test_review(ages=(30, 28), frame_sampler=sampler):
    return review_media(
        "source.mp4",
        face_ages=lambda path: list(ages),
        frame_sampler=frame_sampler,
    )


check("adult faces pass safety check", test_review(ages=(28, 30))["passed"], True)
check("underage face in source is rejected", test_review(ages=(17,))["passed"], False)
check("no face detected is rejected", test_review(ages=())["passed"], False)

# 5. Test CloningRecord
record = CloningRecord(rights_holder="Studio X", all_people_adults=True)
check("cloning record default valid", record.all_people_adults, True)

try:
    CloningRecord(all_people_adults=False)
    check("underage record rejected", False, True)
except ValueError:
    check("underage record rejected", True, True)

print("=" * 60)
print(f"Results: {results['passed']} passed, {results['failed']} failed")
print("=" * 60)
raise SystemExit(1 if results["failed"] else 0)
