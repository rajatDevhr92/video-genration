#!/usr/bin/env python3

import argparse
import json
import uuid

from app.config import OUTPUT_DIR
from app.reference_manager import ReferenceManager
from app.scene_manager import SceneManager


def load_story_plan(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate character references and scene keyframes."
    )

    parser.add_argument(
        "--job-id",
        default=None,
        help="Existing/new job ID.",
    )

    parser.add_argument(
        "--story-plan",
        required=True,
        help="Path to story_plan.json.",
    )

    parser.add_argument(
        "--start-scene",
        type=int,
        default=1,
        help="First scene to generate.",
    )

    parser.add_argument(
        "--end-scene",
        type=int,
        default=None,
        help="Last scene to generate.",
    )

    args = parser.parse_args()

    job_id = args.job_id or uuid.uuid4().hex[:12]

    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    story_plan = load_story_plan(args.story_plan)

    # Save a copy of the story plan inside the job.
    job_story_plan = job_dir / "story_plan.json"

    with open(job_story_plan, "w", encoding="utf-8") as f:
        json.dump(
            story_plan,
            f,
            indent=2,
            ensure_ascii=False,
        )

    characters = story_plan.get("characters", [])

    if not characters:
        raise RuntimeError(
            "Story plan contains no characters."
        )

    print()
    print("=" * 80)
    print("STORY JOB")
    print("=" * 80)
    print("Job ID:", job_id)
    print("Characters:", len(characters))
    print("Scenes:", len(story_plan.get("scenes", [])))

    # ------------------------------------------------------------
    # Generate character references dynamically
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("CHARACTER REFERENCE GENERATION")
    print("=" * 80)

    reference_manager = ReferenceManager()

    references = reference_manager.generate_character_references(
        job_id=job_id,
        characters=characters,
    )

    print()
    print("=" * 80)
    print("CHARACTER REFERENCES READY")
    print("=" * 80)

    for character in characters:
        character_id = character["id"]

        print(
            f"{character.get('name', character_id)} -> "
            f"{references[character_id]}"
        )

    # ------------------------------------------------------------
    # Generate scene keyframes dynamically
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("SCENE KEYFRAME GENERATION")
    print("=" * 80)

    scene_manager = SceneManager()

    results = scene_manager.generate_keyframes(
        job_id=job_id,
        story_plan=story_plan,
        references=references,
        start_scene=args.start_scene,
        end_scene=args.end_scene,
    )

    print()
    print("=" * 80)
    print("KEYFRAME GENERATION COMPLETED")
    print("=" * 80)
    print("Job ID:", job_id)

    for result in results:
        print(
            f"{result['scene_id']} -> "
            f"{result['keyframe']}"
        )


if __name__ == "__main__":
    main()