import json
from pathlib import Path

from .character_reference_generator import (
    CharacterReferenceGenerator,
)


class ReferenceManager:

    def __init__(self):
        self.character_generator = (
            CharacterReferenceGenerator()
        )

    def generate_character_references(
        self,
        job_id,
        characters,
    ):
        references = {}

        for index, character in enumerate(
            characters,
            start=1,
        ):
            character_id = character["id"]

            print()
            print("=" * 70)
            print(
                f"CHARACTER REFERENCE "
                f"{index}/{len(characters)}"
            )
            print("=" * 70)

            print(
                "Character:",
                character.get("name"),
            )

            result = (
                self.character_generator.generate(
                    job_id=job_id,
                    character=character,
                    seed=(
                        1000003
                        + index * 7919
                    ),
                )
            )

            references[character_id] = (
                result["path"]
            )

            print(
                "Reference:",
                result["path"],
            )

        output = Path(
            f"/workspace/story-video-engine/"
            f"output/{job_id}/references/"
            f"character_references.json"
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                references,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return references