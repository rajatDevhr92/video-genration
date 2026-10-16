import json
import os
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class AudioPlanner:

    def __init__(self):
        if OpenAI is None:
            raise RuntimeError(
                "openai package is not installed."
            )

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=api_key
        )

        self.model = os.getenv(
            "STORY_PLANNER_MODEL",
            "gpt-5.6",
        )

    def enrich_plan(self, plan_path):
        plan_path = Path(plan_path)

        with open(
            plan_path,
            "r",
            encoding="utf-8",
        ) as f:
            plan = json.load(f)

        scene_payload = []

        for scene in plan["scenes"]:
            scene_payload.append({
                "id": scene["id"],
                "order": scene["order"],
                "duration": scene["duration"],
                "description": scene["description"],
                "action": scene["action"],
                "character_ids": scene["character_ids"],
                "lighting": scene["lighting"],
                "continuity": scene.get(
                    "continuity",
                    {},
                ),
            })

        prompt = f"""
You are the audio director for a cinematic story.

Create an audio plan for the EXISTING story plan below.

IMPORTANT:
- Do not change the story.
- Do not change scene order.
- Do not change characters.
- Do not change actions.
- Do not change continuity.
- Return ONLY valid JSON.
- Keep narration concise.
- Only add dialogue when supported by the story.
- Do not invent major plot information.
- Keep the same voice assigned to a character across scenes.

VOICE MAPPING RULES:
- Narrator: alloy
- Female characters: nova (or coral/shimmer)
- Male characters: onyx (or echo/ash)
- ALWAYS match the character's voice to their gender (e.g. use female voices nova/coral for women, use male voices onyx/echo for men). Never assign male voices to female characters.

For every scene create:

- narration
- dialogue
- sound_effects
- music

Narration should be empty when unnecessary.

Dialogue should be empty when the story does not support dialogue.

Sound effects should describe things actually occurring or clearly
present in the scene.

Music should specify only mood and intensity.

CHARACTERS:
{json.dumps(
    plan.get("characters", []),
    ensure_ascii=False,
    indent=2,
)}

SCENES:
{json.dumps(
    scene_payload,
    ensure_ascii=False,
    indent=2,
)}

Return exactly:

{{
  "scenes": [
    {{
      "scene_id": "scene_001",

      "narration": [
        {{
          "text": "short narration",
          "voice": "alloy"
        }}
      ],

      "dialogue": [
        {{
          "character_id": "character_001",
          "text": "short spoken line",
          "voice": "nova"
        }}
      ],

      "sound_effects": [
        "heavy rain",
        "footsteps",
        "distant thunder"
      ],

      "music": {{
        "mood": "dark cinematic suspense",
        "intensity": "medium"
      }}
    }}
  ]
}}
"""

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        text = response.output_text

        if not text:
            raise RuntimeError(
                "Audio planner returned empty output."
            )

        try:
            audio_plan = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Audio planner returned invalid JSON:\n"
                + text
            ) from exc

        by_scene = {
            item["scene_id"]: item
            for item in audio_plan.get(
                "scenes",
                []
            )
            if isinstance(item, dict)
            and item.get("scene_id")
        }

        for scene in plan["scenes"]:
            scene_audio = by_scene.get(
                scene["id"],
                {
                    "narration": [],
                    "dialogue": [],
                    "sound_effects": [],
                    "music": {
                        "mood": "",
                        "intensity": "low",
                    },
                },
            )
            # Ensure narration or dialogue is non-empty for every scene
            narr = scene_audio.get("narration", [])
            dial = scene_audio.get("dialogue", [])
            if not narr and not dial:
                fallback_text = (scene.get("description") or scene.get("action") or "").strip()
                if fallback_text:
                    scene_audio["narration"] = [{
                        "text": fallback_text,
                        "voice": "alloy"
                    }]
            scene["audio"] = scene_audio

        with open(
            plan_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                plan,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return plan