"""
Run from the repo root: python -m cloning.cloning_test
Uses fake detectors, so it needs no GPU, models or ffmpeg.
"""
from pydantic import ValidationError

from cloning.cloning_record import CloningRecord
from cloning.cloning_safety import review_media

results = {"passed": 0, "failed": 0}


def check(name, actual, expected):
    if actual == expected:
        results["passed"] += 1
        print(f"PASS  {name}")
    else:
        results["failed"] += 1
        print(f"FAIL  {name}: got {actual!r}, expected {expected!r}")


def sampler(video, output_dir):
    return ["frame_1.png", "frame_2.png"]


def review(nsfw=0.01, ages=(35,), character_ages=(30,), frame_sampler=sampler):
    def face_ages(path):
        return list(character_ages if path == "character.png" else ages)

    return review_media(
        "source.mp4",
        "character.png",
        nsfw_score=lambda path: nsfw,
        face_ages=face_ages,
        frame_sampler=frame_sampler,
    )


def broken(*args):
    raise RuntimeError("model missing")


check("clean media passes", review()["passed"], True)
check("all frames are checked", review()["frames_checked"], 2)
check("sexual content is rejected", review(nsfw=0.9)["passed"], False)
check("score at the limit is rejected", review(nsfw=0.30)["passed"], False)
check("young face in source is rejected", review(ages=(16,))["passed"], False)
check("face under the margin is rejected", review(ages=(22,))["passed"], False)
check("young character is rejected", review(character_ages=(15,))["passed"], False)
check("one young face among adults is rejected", review(ages=(40, 12))["passed"], False)
check("no face in source is rejected", review(ages=())["passed"], False)
check("no face in character is rejected", review(character_ages=())["passed"], False)
check("no frames is rejected", review(frame_sampler=lambda v, o: [])["passed"], False)
check("unreadable video is rejected", review(frame_sampler=broken)["passed"], False)
check(
    "broken detector is rejected",
    review_media(
        "source.mp4",
        "character.png",
        nsfw_score=broken,
        face_ages=lambda path: [30],
        frame_sampler=sampler,
    )["passed"],
    False,
)

GOOD_RECORD = {
    "source_origin": "team_footage",
    "rights_holder": "Example Studio",
    "character_origin": "ai_generated",
    "submitted_by": "Team Member",
    "people": ["Actor One"],
    "consent_reference": "consent/actor_one.pdf",
    "no_sexual_content": True,
    "all_people_adults": True,
}


def record_ok(**changes):
    try:
        CloningRecord(**{**GOOD_RECORD, **changes})
        return True
    except ValidationError:
        return False


check("complete record is accepted", record_ok(), True)
check("sexual content is not accepted", record_ok(no_sexual_content=False), False)
check("non-adults are not accepted", record_ok(all_people_adults=False), False)
check("real people need names", record_ok(people=[]), False)
check("real people need consent", record_ok(consent_reference=" "), False)
check("unknown origin is not accepted", record_ok(source_origin="internet"), False)
check(
    "fully AI media needs no consent file",
    record_ok(source_origin="ai_generated", people=[], consent_reference=""),
    True,
)

print(f"Results: {results['passed']} passed, {results['failed']} failed")
raise SystemExit(1 if results["failed"] else 0)
