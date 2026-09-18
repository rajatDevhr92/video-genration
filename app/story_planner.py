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
    ) -> Dict[str, Any]:

        if not story or not story.strip():

            raise ValueError(
                "Story cannot be empty."
            )

        base_prompt = self._build_prompt(
            story
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
    ) -> str:

        return f"""
You are a professional cinematic story-to-video planner.

Convert the supplied story into ONE structured JSON production plan.

The plan will be used by:

1. Krea 2 for character references.
2. Krea 2 for scene composition.
3. A continuity/keyframe image pipeline.
4. Wan 2.2 for animation.

============================================================
ABSOLUTE OUTPUT RULES
============================================================

Return ONLY valid JSON.

No Markdown.
No code fences.
No explanations.

The JSON MUST be complete.

Do NOT stop early.

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

If the story contains multiple clothing states, choose ONE
representative outfit for the character-reference image.

The chosen outfit should be the character's most visually
important or most frequently used outfit in the story.

ALL OTHER CLOTHING CHANGES MUST EXIST ONLY IN SCENE DATA.

The character object must NOT describe clothing changes.

The character object's clothing field should be short,
specific and visually unambiguous.

Examples:

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

Each scene must contains:

- id
- order
- duration
- location_id
- character_ids
- description
- action
- camera
- lighting
- video_prompt
- audio

Optionally include:

- continuity

Do NOT make these fields unnecessarily long.

============================================================
SCENE DESCRIPTION & ACTION TRANSITIONS
============================================================

Describe only what is physically visible.

Keep each scene description concise (approximately 80 words).

If characters change body posture or spatial position between scenes (e.g. moving from standing to lying down, turning around, crawling over each other, or rotating position), state their exact new body orientation clearly so keyframe and animation pipelines maintain body integrity and smooth motion.

If the scene is sexual, describe the sexual action explicitly.

============================================================
ACTION
============================================================

Describe the main visible action clearly and continuously.

Maximum target:

approximately 50 words.

If the scene is sexual, describe the sexual action explicitly.

============================================================
CAMERA
============================================================

Use a concise actionable camera description.

Maximum target:

approximately 35 words.

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

Every scene MUST contain an "audio" object.

"audio" MUST NEVER be a string.

The structure MUST be:

"audio": {{
  "narration": [],
  "dialogue": [],
  "sound_effects": [],
  "music": {{
    "mood": "",
    "intensity": ""
  }}
}}

"narration" MUST be an array of objects.

Each narration object MUST contain:
- text
- voice

Example:

"narration": [
  {{
    "text": "Narration text.",
    "voice": "alloy"
  }}
]

If there is no narration:

"narration": []

"dialogue" MUST be an array of objects.

Each dialogue object MUST contain:
- character_id
- text

The character_id MUST exactly match one of the character IDs
defined in the characters array.

Example:

"dialogue": [
  {{
    "character_id": "character_001",
    "text": "Hello."
  }}
]

If there is no dialogue:

"dialogue": []

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
- Each character's "clothing" field must contain ONE canonical outfit only.
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
  "narration": [],
  "dialogue": [],
  "sound_effects": [],
  "music": {{
    "mood": "",
    "intensity": ""
  }}
}}

The "audio" value MUST be an object, never a string.

"narration" must be a list.

"dialogue" must be a list.

"sound_effects" must be a list.

"music" must be an object.

Every dialogue character_id must reference an existing
character ID.

Do not omit audio from any scene.
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