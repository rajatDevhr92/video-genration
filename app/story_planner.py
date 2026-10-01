from typing import Dict, Any


class StoryPlanValidationError(ValueError):

    def __init__(
        self,
        message,
        missing_fields=None,
        details=None,
    ):
        super().__init__(message)

        self.missing_fields = (
            missing_fields or []
        )

        self.details = details


class StoryPlanner:

    """
    Converts a story into a structured production plan.

    The plan is designed for a cinematic image/video pipeline.

    Important design rule:
    Keep generated JSON compact enough that the local LLM can
    reliably complete the entire document.
    """

    REQUIRED_PLAN_FIELDS = [
        "title",
        "summary",
        "characters",
        "locations",
        "scenes",
    ]

    REQUIRED_CHARACTER_FIELDS = [
        "id",
        "name",
        "appearance",
        "clothing",
        "personality",
        "visual_anchor",
    ]

    REQUIRED_SCENE_FIELDS = [
        "id",
        "order",
        "duration",
        "location_id",
        "character_ids",
        "clothing_state",
        "pose_type",
        "penetration_state",
        "primary_motion",
        "description",
        "action",
        "camera",
        "lighting",
        "video_prompt",
        "audio",
    ]

    def __init__(self, provider):

        self.provider = provider

    def build(
        self,
        story: str,
        target_scenes: int = None,
    ) -> Dict[str, Any]:

        if not story or not story.strip():

            raise ValueError(
                "Story cannot be empty."
            )

        base_prompt = self._build_prompt(
            story,
            target_scenes=target_scenes,
        )

        print(
            "[DEBUG] Calling StoryPlanner..."
        )

        result = self.provider.generate_json(
            base_prompt
        )

        try:

            self._validate(result)

            return result

        except ValueError as first_error:

            print(
                "[PLANNER] First plan failed validation."
            )

            print(
                f"[PLANNER] Error: {first_error}"
            )

            retry_prompt = (
                self._build_retry_prompt(
                    story=story,
                    result=result,
                    error=str(first_error),
                )
            )

            print(
                "[DEBUG] Calling StoryPlanner retry..."
            )

            retry_result = (
                self.provider.generate_json(
                    retry_prompt
                )
            )

            try:

                self._validate(
                    retry_result
                )

                return retry_result

            except ValueError as second_error:

                raise StoryPlanValidationError(
                    message=(
                        "The story planner returned "
                        "an incomplete or invalid "
                        "production plan."
                    ),
                    missing_fields=(
                        self._find_missing_fields(
                            retry_result
                        )
                    ),
                    details=str(
                        second_error
                    ),
                ) from second_error

    def _build_prompt(
        self,
        story: str,
        target_scenes: int = None,
    ) -> str:

        # Auto-detect pre-numbered scenes (e.g., "Scene 1 [", "Scene 2 [", etc.)
        import re
        detected_scenes = re.findall(r"Scene\s+(\d+)\s*\[", story, flags=re.IGNORECASE)
        num_detected = len(detected_scenes)

        if target_scenes:
            scene_directive = f"""
============================================================
MANDATORY TARGET SCENE COUNT DIRECTIVE
============================================================
- You MUST divide this story into EXACTLY {target_scenes} sequential scenes.
- Do NOT condense or merge the story into fewer scenes.
- Break down every major narrative beat into its own distinct scene object (order 1 through {target_scenes}).
"""
        elif num_detected > 0:
            scene_directive = f"""
============================================================
MANDATORY PRE-STRUCTURED SCENE MATCHING DIRECTIVE ({num_detected} SCENES)
============================================================
- The input story is already formatted into EXACTLY {num_detected} explicit scene blocks (Scene 1 through Scene {num_detected}).
- You MUST generate EXACTLY {num_detected} scene objects in your JSON output (order 1 through {num_detected}).
- DO NOT condense, merge, or drop any of the {num_detected} scenes!
- Map each "Scene X [Shot Type, 5s]" directly into its corresponding JSON scene object with order = X and duration = 5.
"""
        else:
            scene_directive = """
============================================================
SCENE COUNT & FULL STORY COVERAGE DIRECTIVE
============================================================
- Do NOT over-compress the story.
- Generate enough scenes to fully cover all events described in the story without skipping scenes or rushing through plot points.
"""

        return f"""
{scene_directive}

============================================================
STORY FIDELITY
============================================================

Preserve the actual story.

Do not invent major plot events.

Do not remove important story events.

Do not change relationships.

Do not introduce unnecessary characters.

Do not invent unnecessary locations.

Only add visual details that are reasonable and do not
contradict the supplied story.

============================================================
SEXUAL / ADULT CONTENT
============================================================
If the original story contains sexual activity, nudity,
or adult erotic situations, you MUST preserve them explicitly.

Rules:
- Do NOT sanitize, soften, or remove sexual content.
- Do NOT replace explicit acts with vague language such as
  "they make love", "intimate moment", "passionate encounter".
- Write the sexual acts clearly and physically in the
  scene "description", "action", and "video_prompt" fields.
- Use direct language when the story requires it
  (penetration, oral sex, handjob, thrusting, nudity,
  body parts, wetness, moans, etc.).
- Keep the tone cinematic and physical, not pornographic
  purple prose, but still explicit enough for the image
  and video models to understand the exact act.
- All characters remain adults (18+). Never create minors.
- Clothing removal or nudity that occurs in the story
  must be reflected in the relevant scene descriptions.
- The character "clothing" field still remains ONE
  canonical outfit only. Scene-specific nudity or
  clothing changes belong only in the scene data.

============================================================
CHARACTERS
============================================================

Create ONE character object for EACH important person.

CRITICAL:

ONE CHARACTER OBJECT = ONE PERSON.

Never merge two people into one character.

Never create multiple visual versions of the same person.

Never create a character sheet.

Every character must represent exactly one individual.

ALL CHARACTERS MUST BE ADULTS.

Never assign a minor, teenage, underage, child, or adolescent age.

If the story does not specify an age, use an adult age of 18+
appropriate to the story.

For each character provide:

- id
- name
- appearance
- clothing
- personality
- visual_anchor

Keep these fields concise but visually specific.

The appearance must establish a stable identity:

- adult age
- gender/presentation when relevant
- skin tone
- face
- eyes
- hair
- hairstyle
- body/build
- distinctive features

Do NOT use generic stereotypes.

The same character must look like the same person
in every scene.

============================================================
CHARACTER CLOTHING RULE
============================================================

The "clothing" field is used DIRECTLY to generate the
character's ONE character-reference image.

Therefore the field MUST describe exactly ONE outfit.

CRITICAL:

ONE CHARACTER
=
ONE PERSON
=
ONE CANONICAL OUTFIT

The "clothing" field MUST NOT contain multiple outfits.

NEVER write:

"black swim trunks, white towel gown"

NEVER write:

"red bikini, white robe"

NEVER write:

"suit, later jacket"

NEVER write:

"casual clothes / formal clothes"

NEVER write:

"initially X, later Y"

NEVER write:

"X, then changes to Y"

NEVER write:

"X when outside and Y indoors"

CRITICAL NUDITY & CLOTHING RULE:
- IF THE CHARACTER IS NUDE, UNCLOTHED, OR NAKED IN THE STORY, YOU MUST SET "clothing": "Fully nude, completely unclothed, bare skin".
- DO NOT invent bras, lingerie, thongs, underwear, swimwear, or casual clothes when the story specifies the character is nude or unclothed!
- If the character wears an outfit for most of the story, set "clothing" to that ONE canonical outfit.
- If the story contains multiple clothing states, choose ONE representative outfit (or "Fully nude, completely unclothed" if nude).

ALL OTHER CLOTHING CHANGES MUST EXIST ONLY IN SCENE DATA.

The character object must NOT describe clothing changes.

The character object's clothing field should be short,
specific and visually unambiguous.

Examples:

GOOD (for nude story):
"Fully nude, completely unclothed, bare skin"

GOOD:
"Black swim trunks"

GOOD:
"Red bikini with thin straps"

GOOD:
"Light blue cotton dress with white sandals"

BAD:
"Black swim trunks and white towel gown"

BAD:
"Red bikini, later white towel gown"

BAD:
"Black swim trunks initially, then towel gown"

BAD:
"Casual clothing with a formal outfit for later scenes"

The character-reference image must therefore have exactly:

ONE person.
ONE outfit.
ONE pose.
ONE camera.
ONE continuous full-body frame.

============================================================
CHARACTER VISUAL ANCHOR
============================================================

The visual_anchor is extremely important.

It will be used by the image-generation pipeline.

It must contain only the most important stable visual
identity information.

Example:

"Adult man, short black hair, brown eyes, olive skin,
lean athletic build, small scar above left eyebrow."

Do NOT put scene-specific actions into visual_anchor.

Do NOT put multiple people into visual_anchor.

Do NOT put alternate outfits into visual_anchor.

============================================================
LOCATIONS
============================================================

Create only locations actually required by the story.

Each location contains:

- id
- name
- description
- visual_style

Keep descriptions concise.

============================================================
SCENES
============================================================

Create the necessary cinematic scenes.

Each scene must contain:

- id
- order
- duration
- location_id
- character_ids
- clothing_state (e.g. "Fully nude, completely unclothed, bare skin")
- pose_type (exact anatomical posture, e.g. "On all fours in doggy position", "Standing upright")
- penetration_state (exact intimate contact or penetration state, e.g. "Male penis entering female pussy from behind", "None")
- primary_motion (isolated kinetic movement description)
- description
- action
- camera (MUST combine technical cinematic camera framing with turnaround sheet panel tag: e.g., "Medium close-up, low angle, front-facing on bed (matches LEFT PANEL Front View of turnaround sheet)")
- lighting
- video_prompt
- audio

For scenes starting from scene 2 (order >= 2), you MUST also include:
- previous_scene_context (1-3 short sentences describing the previous scene's exact body positions, clothing state, and key contact/genital alignment for seamless visual continuity)

Optionally include:

- continuity

Do NOT make these fields unnecessarily long.

============================================================
SCENE DESCRIPTION & ACTION TRANSITIONS
============================================================

Describe only what is physically visible.

Keep each scene description concise (approximately 80 words).

If characters change body posture or spatial position between scenes (e.g. moving from standing to lying down, turning around, crawling over each other, or rotating position), state their exact new body orientation clearly so keyframe and animation pipelines maintain body integrity and smooth motion.

CRITICAL HAND POSITION RULE:
- In "description", "action", and "previous_scene_context", ALWAYS explicitly state the exact position of both hands for every character (e.g. "both hands firmly gripping her hips", "left hand resting flat on mattress", "right hand reaching under to rub her clit").
- Never leave hands unmentioned or floating in ambiguity.

If the scene is sexual, describe the sexual action explicitly.

============================================================
ACTION
============================================================

Describe the main visible action clearly and continuously.

State exact limb placement, torso posture, and hand anchoring points.

Maximum target:

approximately 50 words.

If the scene is sexual, describe the sexual action explicitly.

============================================================
CAMERA
============================================================

The camera description MUST combine full technical camera details with a turnaround panel tag.

Requirements:
1. Technical Camera Specs: State the exact shot size (Wide Shot, Medium Shot, Close-up, Extreme Close-up), camera height/angle (Low angle, Eye-level, High angle, Over-the-shoulder), and focus point/framing.
2. Turnaround Panel Tag: Indicate which panel of the character 3-panel turnaround sheet to use for reference:
   - Front-facing shots / facing camera → "(matches LEFT PANEL Front View of turnaround sheet)"
   - Side / profile / 3/4 angle shots → "(matches CENTER PANEL 3/4 Side View of turnaround sheet)"
   - Rear / back / doggy-style shots from behind → "(matches RIGHT PANEL Rear View of turnaround sheet)"

CRITICAL RULE:
NEVER output only the panel tag! Always combine detailed technical camera framing with the matching panel tag.

Examples:
- GOOD: "Medium shot, low angle from behind, tight focus on hips and rear alignment (matches RIGHT PANEL Rear View of turnaround sheet)"
- GOOD: "Close-up, front-facing eye-level, tracking face and chest expressions (matches LEFT PANEL Front View of turnaround sheet)"
- GOOD: "Wide shot, 3/4 side angle, low position capturing profile on bed (matches CENTER PANEL 3/4 Side View of turnaround sheet)"
- BAD: "(matches LEFT PANEL Front View of turnaround sheet)"

Maximum target:
approximately 45 words.

============================================================
LIGHTING
============================================================

Use a concise actionable lighting description.

Maximum target:

approximately 30 words.

============================================================
VIDEO PROMPT
============================================================

Describe only the movement required for animation.

Maximum target:

approximately 60 words.

If the scene is sexual, describe the sexual action explicitly

CRITICAL AUDIO VOICE DIRECTIVE:
- Every scene MUST contain active voice audio text in either "narration" OR "dialogue".
- If characters are speaking, gasping, or vocalizing in the scene: populate "dialogue" with character objects:
  "dialogue": [
    {{
      "character_id": "character_001",
      "text": "Don't stop..."
    }}
  ]
- If characters are not speaking aloud: populate "narration" with a voiceover object:
  "narration": [
    {{
      "text": "Descriptive voiceover narration of the scene atmosphere or character thoughts...",
      "voice": "alloy"
    }}
  ]
- NEVER leave both "narration": [] and "dialogue": [] empty at the same time in any scene! Every scene MUST have voice text to generate audio TTS!

"sound_effects" MUST be an array of short strings.

Example:

"sound_effects": [
  "water splashing",
  "light laughter"
]

If there are no sound effects:

"sound_effects": []

"music" MUST always be an object:

"music": {{
  "mood": "playful",
  "intensity": "low"
}}

Do NOT return:

"audio": "some text"

Do NOT return:

"audio": "dialogue..."

Do NOT return audio as plain text.

Keep dialogue short.

Do not generate long speeches.

Do not repeat dialogue unnecessarily.

============================================================
CONTINUITY
============================================================

Continuity should only contain information necessary
to preserve visual consistency between scenes.

Do not repeat the complete scene description.

Do not repeat the complete character description.

Do not repeat the complete location description.

============================================================
IMPORTANT COMPRESSION RULE
============================================================

The complete JSON must remain compact.

Do NOT pad the response.

Do NOT repeat information.

Do NOT use unnecessary prose.

Do NOT create huge dialogue blocks.

Do NOT create repeated continuity notes.

The goal is a complete, concise production plan.

============================================================
REQUIRED JSON STRUCTURE
============================================================

{{
  "title": "...",
  "summary": "...",
  "characters": [],
  "locations": [],
  "scenes": []
}}

Required character fields:

{self.REQUIRED_CHARACTER_FIELDS}

Required scene fields:

{self.REQUIRED_SCENE_FIELDS}

============================================================
ORIGINAL STORY
============================================================

{story}

============================================================
FINAL INSTRUCTION
============================================================

Generate the COMPLETE production plan now.

Return ONLY the JSON object.
"""

    def _build_retry_prompt(
        self,
        story: str,
        result: Any,
        error: str,
    ) -> str:

        return f"""
The previous production plan was invalid.

Validation error:

{error}

Generate the COMPLETE production plan again.

IMPORTANT:

- If the original story is sexual, keep the sexual content
  explicit in description, action, and video_prompt.
- Do not sanitize adult content.
- Return ONLY valid JSON.
- Do not use Markdown.
- Do not use code fences.
- Do not write explanations.
- Do not truncate the response.
- Do not omit scenes.
- Do not omit required fields.
- Keep descriptions concise.
- Keep dialogue concise.
- Do not repeat information.
- ONE character object represents ONE adult person.
- ALL characters must be adults.
- Do not create minor or teenage characters.
- Each character's "clothing" field must contain ONE canonical outfit only (or "Fully nude, completely unclothed, bare skin" if nude/naked in story). Never invent bras or thongs when nude.
- Never put outfit changes inside the character's "clothing" field.
- Never use "initially", "later", "eventually", "then changes to",
  or similar transition language inside character clothing.
- Put scene-specific clothing changes only in scene descriptions.
- Preserve the original story.
- Preserve character identities.
- Preserve scene continuity.

Required top-level fields:

{self.REQUIRED_PLAN_FIELDS}

Required character fields:

{self.REQUIRED_CHARACTER_FIELDS}

Required scene fields:

{self.REQUIRED_SCENE_FIELDS}

Previous result:

{result}

Original story:

{story}

Return the COMPLETE corrected JSON now.

AUDIO REQUIREMENT:

Every scene MUST contain:
"audio": {{
  "narration": [
    {{
      "text": "Cinematic voiceover narration text...",
      "voice": "alloy"
    }}
  ],
  "dialogue": [],
  "sound_effects": ["soft breathing"],
  "music": {{
    "mood": "sensual",
    "intensity": "medium"
  }}
}}

CRITICAL VOICE RULE: Either "narration" OR "dialogue" MUST contain spoken/voice text in every scene. NEVER leave both "narration": [] and "dialogue": [] empty simultaneously!
"""

    def _validate(
        self,
        result: Any,
    ):

        if not isinstance(
            result,
            dict,
        ):

            raise ValueError(
                "Plan must be a JSON object."
            )

        for field in self.REQUIRED_PLAN_FIELDS:

            if field not in result:

                raise ValueError(
                    f"Missing top-level field: {field}"
                )

        if not isinstance(
            result["characters"],
            list,
        ):

            raise ValueError(
                "'characters' must be a list."
            )

        if not isinstance(
            result["locations"],
            list,
        ):

            raise ValueError(
                "'locations' must be a list."
            )

        if not isinstance(
            result["scenes"],
            list,
        ):

            raise ValueError(
                "'scenes' must be a list."
            )

        if not result["scenes"]:

            raise ValueError(
                "At least one scene is required."
            )

        # --------------------------------------------------
        # Validate characters
        # --------------------------------------------------

        character_ids = set()

        for character in result["characters"]:

            if not isinstance(
                character,
                dict,
            ):

                raise ValueError(
                    "Character must be an object."
                )

            for field in (
                self.REQUIRED_CHARACTER_FIELDS
            ):

                if field not in character:

                    raise ValueError(
                        "Character missing field: "
                        f"{field}"
                    )

            character_id = character["id"]

            if character_id in character_ids:

                raise ValueError(
                    f"Duplicate character id: "
                    f"{character_id}"
                )

            character_ids.add(
                character_id
            )

        # --------------------------------------------------
        # Validate scenes
        # --------------------------------------------------

        scene_ids = set()

        for scene in result["scenes"]:

            if not isinstance(
                scene,
                dict,
            ):

                raise ValueError(
                    "Scene must be an object."
                )

            # --------------------------------------------------
            # Auto-populate keyframe-enrichment scene fields if missing
            # --------------------------------------------------
            if not scene.get("clothing_state"):
                desc_lower = f"{scene.get('description', '')} {scene.get('action', '')}".lower()
                if any(w in desc_lower for w in ["nude", "naked", "unclothed", "bare skin", "undress", "strips"]):
                    scene["clothing_state"] = "Fully nude, completely unclothed, bare skin"
                else:
                    scene["clothing_state"] = "Canonical outfit from character reference"

            if not scene.get("pose_type"):
                scene["pose_type"] = (scene.get("action") or scene.get("description") or "Standing upright")[:80]

            if not scene.get("penetration_state"):
                desc_lower = f"{scene.get('description', '')} {scene.get('action', '')}".lower()
                if any(w in desc_lower for w in ["penetrat", "doggystyle", "thrust", "cock inside", "pussy contact", "sex", "fuck"]):
                    scene["penetration_state"] = "Active intimate penetration / body contact"
                else:
                    scene["penetration_state"] = "None"

            if not scene.get("primary_motion"):
                scene["primary_motion"] = (scene.get("video_prompt") or scene.get("action") or "")[:80]

            scene_order = scene.get("order", 1)
            if scene_order >= 2 and not scene.get("previous_scene_context"):
                scenes_list = result.get("scenes", [])
                prev_idx = scene_order - 2
                if 0 <= prev_idx < len(scenes_list):
                    prev_scene = scenes_list[prev_idx]
                    prev_desc = prev_scene.get("description", "")
                    prev_clothing = prev_scene.get("clothing_state", "Fully nude, completely unclothed, bare skin")
                    prev_pose = prev_scene.get("pose_type", prev_scene.get("action", ""))
                    scene["previous_scene_context"] = f"In scene {scene_order - 1}, character was in pose ({prev_pose}). Clothing state: {prev_clothing}. Context: {prev_desc[:120]}"

            for field in (
                self.REQUIRED_SCENE_FIELDS
            ):

                if field not in scene:

                    raise ValueError(
                        "Scene missing field: "
                        f"{field}"
                    )

            scene_id = scene["id"]

            if scene_id in scene_ids:

                raise ValueError(
                    f"Duplicate scene id: "
                    f"{scene_id}"
                )

            scene_ids.add(
                scene_id
            )

            # --------------------------------------------------
            # Validate audio
            # --------------------------------------------------

            audio = scene.get("audio")

            if not isinstance(audio, dict):
                raise ValueError(
                    f"Scene {scene_id} audio must be an object."
                )

            narration = audio.get(
                "narration",
                [],
            )

            if not isinstance(
                narration,
                list,
            ):
                raise ValueError(
                    f"Scene {scene_id} narration "
                    "must be a list."
                )

            dialogue = audio.get(
                "dialogue",
                [],
            )

            if not isinstance(
                dialogue,
                list,
            ):
                raise ValueError(
                    f"Scene {scene_id} dialogue "
                    "must be a list."
                )

            sound_effects = audio.get(
                "sound_effects",
                [],
            )

            if not isinstance(
                sound_effects,
                list,
            ):
                raise ValueError(
                    f"Scene {scene_id} sound_effects "
                    "must be a list."
                )

            music = audio.get(
                "music",
                {},
            )

            if not isinstance(
                music,
                dict,
            ):
                raise ValueError(
                    f"Scene {scene_id} music "
                    "must be an object."
                )

            for item in narration:

                if not isinstance(
                    item,
                    dict,
                ):
                    raise ValueError(
                        f"Scene {scene_id} narration "
                        "items must be objects."
                    )

                if not item.get("text"):
                    raise ValueError(
                        f"Scene {scene_id} narration "
                        "item missing text."
                    )

            for item in dialogue:

                if not isinstance(
                    item,
                    dict,
                ):
                    raise ValueError(
                        f"Scene {scene_id} dialogue "
                        "items must be objects."
                    )

                if not item.get("character_id"):
                    raise ValueError(
                        f"Scene {scene_id} dialogue "
                        "item missing character_id."
                    )

                if not item.get("text"):
                    raise ValueError(
                        f"Scene {scene_id} dialogue "
                        "item missing text."
                    )

                if item["character_id"] not in character_ids:
                    raise ValueError(
                        f"Scene {scene_id} dialogue references "
                        f"unknown character: "
                        f"{item['character_id']}"
                    )

            # --------------------------------------------------
            # Voice Audio Enforcement & Auto-Repair:
            # Every scene MUST contain active voice audio. If both narration
            # and dialogue are empty, auto-populate narration from scene description.
            # --------------------------------------------------
            if not narration and not dialogue:
                fallback_text = (scene.get("description") or scene.get("action") or "").strip()
                if fallback_text:
                    narration = [{
                        "text": fallback_text,
                        "voice": "alloy"
                    }]
                    audio["narration"] = narration

            for effect in sound_effects:

                if not isinstance(
                    effect,
                    str,
                ):
                    raise ValueError(
                        f"Scene {scene_id} sound_effects "
                        "must contain strings."
                    )

            for character_id in scene[
                "character_ids"
            ]:

                if character_id not in character_ids:

                    raise ValueError(
                        f"Scene {scene_id} references "
                        f"unknown character: "
                        f"{character_id}"
                    )

    def _find_missing_fields(
        self,
        result: Any,
    ):

        missing = []

        if not isinstance(
            result,
            dict,
        ):

            return [
                "complete JSON object"
            ]

        for field in self.REQUIRED_PLAN_FIELDS:

            if field not in result:

                missing.append(field)

        characters = result.get(
            "characters",
            [],
        )

        if isinstance(
            characters,
            list,
        ):

            for index, character in enumerate(
                characters
            ):

                if isinstance(
                    character,
                    dict,
                ):

                    for field in (
                        self.REQUIRED_CHARACTER_FIELDS
                    ):

                        if field not in character:

                            missing.append(
                                f"characters[{index}].{field}"
                            )

        scenes = result.get(
            "scenes",
            [],
        )

        if isinstance(
            scenes,
            list,
        ):

            for index, scene in enumerate(
                scenes
            ):

                if isinstance(
                    scene,
                    dict,
                ):

                    for field in (
                        self.REQUIRED_SCENE_FIELDS
                    ):

                        if field not in scene:

                            missing.append(
                                f"scenes[{index}].{field}"
                            )

        return missing