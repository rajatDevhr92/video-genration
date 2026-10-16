import json
import logging
import re
from typing import Dict, Any, List, Optional
from .continuity import ACTS, EVENTS, normalize_gender
from .story_parser import parse_story
from .compiler import compile_scene, enforce_wardrobe
from .facts import extract_facts

logger = logging.getLogger(__name__)

CAST_PROMPT = """Read this story and extract the characters and the primary setting.
Return JSON ONLY with this schema:
{{
  "title": "Story Title",
  "setting": "Description of the primary environment/room",
  "characters": [
    {{
      "id": "character_001",
      "name": "Character Name",
      "gender": "female or male",
      "appearance": "Physical appearance (age, hair, ethnicity, build)",
      "clothing": "Starting outfit ONLY",
      "visual_anchor": "Key facial and physical anchors"
    }}
  ]
}}

Story:
{story}
"""


def build_plan(
    provider: Any,
    story: str,
    target_scenes: int | None = None,
) -> Dict[str, Any]:
    """
    Modular, deterministic pipeline:
    1. Parser (code) -> scenes, durations, shot hints
    2. Cast/setting (1 small LLM call)
    3. Facts per scene (small LLM calls with closed enums & local retries)
    4. Compiler (code) -> video_prompt, camera turnaround panel, pose lock, hands
    5. Wardrobe enforcer (code) -> guarantees bottom removal for exposed acts
    """
    meta, preamble, parsed_scenes = parse_story(story, target_scenes=target_scenes)
    if not parsed_scenes:
        raise ValueError("No scenes could be parsed from the story.")

    print(f"[PLANNER] Parsed {len(parsed_scenes)} scenes from story text.")
    print("[PLANNER] Extracting cast and setting...")
    cast_res = provider.generate_json(CAST_PROMPT.format(story=story))
    if not isinstance(cast_res, dict):
        cast_res = {}

    cast = cast_res.get("characters") or cast_res.get("cast") or []
    if not cast:
        cast = [
            {
                "id": "character_001",
                "name": "Protagonist",
                "gender": "female",
                "appearance": "Adult woman, 28 years old, athletic build",
                "clothing": "Casual outfit",
                "visual_anchor": "Adult woman, athletic build",
            }
        ]

    for i, c in enumerate(cast):
        if not c.get("id"):
            c["id"] = f"character_{i+1:03d}"
        if not c.get("gender") or c.get("gender") not in {"female", "male"}:
            c["gender"] = normalize_gender(
                c.get("appearance", ""),
                c.get("visual_anchor", ""),
                c.get("name", ""),
            )
        if not c.get("visual_anchor"):
            c["visual_anchor"] = f"{c.get('appearance', '')}, {c.get('clothing', '')}"
        if not c.get("personality"):
            c["personality"] = "Expressive and natural"

    cast_by_id = {c["id"]: c for c in cast}
    location_id = "location_001"
    setting_desc = (
        meta.get("setting")
        or cast_res.get("setting")
        or "Cinematic interior environment with warm atmospheric lighting"
    )
    title = meta.get("title") or cast_res.get("title") or "Cinematic Story"

    out = []
    prev_facts = None
    prev_compiled = None

    for sc in parsed_scenes:
        order = sc.get("order", len(out) + 1)
        print(f"[PLANNER] Extracting closed facts for Scene {order}/{len(parsed_scenes)}...")
        facts = extract_facts(provider, sc, cast, prev_facts)
        comp = compile_scene(sc, facts, prev_compiled, cast_by_id, location_id=location_id)
        out.append(comp)
        prev_facts = facts
        prev_compiled = comp

    enforce_wardrobe(out)

    return {
        "title": title,
        "summary": f"{title} - cinematic visual narrative",
        "characters": cast,
        "locations": [
            {
                "id": location_id,
                "name": meta.get("setting", "Main Setting"),
                "description": setting_desc,
                "visual_style": "cinematic photorealistic 8k lighting and composition",
            }
        ],
        "scenes": out,
    }


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
        "gender",
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
        "act",
        "wardrobe_events",
        "clothing_state",
        "pose_type",
        "primary_motion",
        "motion_intensity",
        "expression",
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

        try:
            print("[DEBUG] Running modular build_plan...")
            plan = build_plan(self.provider, story, target_scenes=target_scenes)
            self._validate(plan)
            return plan
        except Exception as modular_err:
            print(
                f"[PLANNER] Modular build_plan failed ({modular_err}). "
                "Falling back to monolithic generation..."
            )

        base_prompt = self._build_prompt(
            story,
            target_scenes=target_scenes,
        )

        print(
            "[DEBUG] Calling StoryPlanner monolithic..."
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
- Clothing removal or clothing changes that occur in the story
  must be expressed through scene wardrobe_events.
- The character "clothing" field represents the outfit the character first appears in.

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
- gender ("female" or "male")
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

CRITICAL CLOTHING RULE:
- "clothing" = the outfit the character first appears in.
- Undressing and clothing changes are expressed only through scene "wardrobe_events".
- If the character starts the story fully nude, set "clothing": "Fully nude, completely unclothed, bare skin". Otherwise, set their initial canonical outfit.

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
- act (REQUIRED — EXACTLY one of: "none", "kiss", "touch", "oral", "vaginal", "anal", "other_contact")
- wardrobe_events (REQUIRED — list of wardrobe change events: [{{"character_id": "...", "event": "remove_top" | "remove_bottom" | "remove_all" | "dress_all"}}], or empty list [] when nobody changes clothes)
- clothing_state (e.g. "Canonical outfit from character reference" or "wearing the top of: ...; lower body bare" or "fully nude, bare skin")
- pose_type (exact anatomical posture, e.g. "Lying down on bed", "Standing upright")
- primary_motion (isolated kinetic movement description — MUST describe actual physical body movement only. NEVER use placeholder tags like motion_*, or any bracketed tokens. NEVER write camera verbs. Write the actual movement in plain English, e.g. "Character moves with steady continuous rhythm, maintaining deliberate physical pacing")
- motion_intensity ("low", "medium", "fast", or "deep and slow")
- expression (facial expression state, e.g. "Woman: eyes closed, soft smile. Man: focused expression")
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
MOTION QUALITY RULES (CRITICAL — READ BEFORE GENERATING)
============================================================

RULE 1 — primary_motion:
- MUST describe ONLY the physical body movement of the characters.
- NEVER use camera verbs (dolly, push, track, zoom, pan, pull).
- NEVER use placeholder tags (motion_*, [motion], etc.).
- Write the exact movement in plain English. Name the character performing it.
- GOOD: "Character moves with steady continuous rhythm, maintaining deliberate physical pacing"
- BAD: "Initiate a continuous rhythmic motion_action"
- BAD: "Track the camera alongside the character"

RULE 2 — motion_intensity:
- Must escalate across the scene arc. DO NOT keep the same value for all scenes.
- Use only these values: "very slow" | "slow" | "medium" | "fast" | "deep and slow" | "rhythmic medium" | "final deep hold"
- For a 15-scene arc: scenes 1-3 = slow, scenes 4-6 = medium, scenes 7-10 = fast, scenes 11-14 = fast, scene 15 = final deep hold

RULE 5 — hand_positions:
- Specify BOTH hands for EVERY character.
- Must be physically consistent with the pose_lock.
- GOOD: "Man: both hands firmly gripping her hips. Woman: both hands planted flat on mattress."

RULE 6 — video_prompt structure:
- Lead with: [Character A] [specific hip/pelvis action] while [Character B] [reaction].
- Then: Exact contact: [genital_alignment].
- Then: Hands: [hand_positions].
- Then: Bodies remain locked in [pose_lock].
- Then: Camera: [camera description with panel tag].

RULE 7 — previous_scene_context (required for order >= 2):
- Describe: exact ending body positions + hand positions + penetration depth + last motion direction.

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

CRITICAL MOTION RULES:
- primary_motion MUST describe actual physical body movement only. NEVER use placeholders like motion_*. NEVER use camera verbs. Write the exact movement in plain English: e.g. "Character moves with steady deliberate pacing, maintaining close physical contact on each movement."
- motion_intensity MUST escalate across scenes. Use: "very slow" | "slow" | "medium" | "fast" | "deep and slow" | "rhythmic medium" | "final deep hold". Do NOT repeat "slow" for every scene.
- act MUST be EXACTLY one of: "none", "kiss", "touch", "oral", "vaginal", "anal", "other_contact".
- wardrobe_events MUST be a list of wardrobe events [{{"character_id": "...", "event": "remove_top|remove_bottom|remove_all|dress_all"}}] (empty list [] if no clothing change).
- video_prompt MUST describe clear physical movement and character reaction matching the scene action.

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

        if "title" not in result or not result["title"]:
            result["title"] = "Cinematic Story Plan"

        if "summary" not in result or not result["summary"]:
            result["summary"] = "Cinematic video production plan."

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

        for idx, character in enumerate(result["characters"], start=1):
            if not isinstance(character, dict):
                raise ValueError("Character must be an object.")

            if not character.get("id"):
                character["id"] = f"char_{idx:03d}"

            if not character.get("name"):
                character["name"] = f"Character {idx}"

            if not character.get("appearance"):
                character["appearance"] = "Adult person, dark hair, expressive eyes, athletic build"

            if not character.get("clothing"):
                character["clothing"] = "Casual everyday outfit"

            if not character.get("personality"):
                character["personality"] = "Passionate and expressive"

            if not character.get("visual_anchor"):
                character["visual_anchor"] = character.get("appearance")

            if not character.get("gender"):
                character["gender"] = normalize_gender(
                    character.get("visual_anchor", ""),
                    character.get("appearance", ""),
                    character.get("name", ""),
                )
            elif character.get("gender") not in {"female", "male"}:
                character["gender"] = normalize_gender(character.get("gender", ""))

            if character.get("gender") not in {"female", "male"}:
                raise ValueError(f"Character {character.get('id')} has invalid gender: {character.get('gender')}. Must be 'female' or 'male'.")

            for field in self.REQUIRED_CHARACTER_FIELDS:
                if field not in character:
                    raise ValueError(f"Character missing field: {field}")

            character_id = character["id"]
            if character_id in character_ids:
                raise ValueError(f"Duplicate character id: {character_id}")
            character_ids.add(character_id)

        # --------------------------------------------------
        # Validate scenes
        # --------------------------------------------------
        scene_ids = set()

        for idx, scene in enumerate(result["scenes"], start=1):
            if not isinstance(scene, dict):
                raise ValueError("Scene must be an object.")

            if not scene.get("id"):
                scene["id"] = f"scene_{idx:03d}"

            import re

            raw_order = scene.get("order")
            if raw_order is None or raw_order == "":
                scene["order"] = idx
            else:
                try:
                    scene["order"] = int(raw_order)
                except (ValueError, TypeError):
                    cleaned_order = re.sub(r"[^\d]", "", str(raw_order))
                    scene["order"] = int(cleaned_order) if cleaned_order else idx

            raw_duration = scene.get("duration")
            if raw_duration is None or raw_duration == "":
                scene["duration"] = 5.0
            else:
                try:
                    scene["duration"] = float(raw_duration)
                except (ValueError, TypeError):
                    cleaned_dur = re.sub(r"[^\d.]", "", str(raw_duration))
                    try:
                        scene["duration"] = float(cleaned_dur) if cleaned_dur else 5.0
                    except ValueError:
                        scene["duration"] = 5.0

            if not scene.get("location_id"):
                scene["location_id"] = (
                    result["locations"][0]["id"] if result.get("locations") else "loc_001"
                )

            if not scene.get("character_ids"):
                scene["character_ids"] = [c["id"] for c in result.get("characters", [])]

            if not scene.get("description"):
                scene["description"] = scene.get("action", "Cinematic scene action.")

            if not scene.get("action"):
                scene["action"] = scene.get("description", "Physical movement.")

            if not scene.get("camera"):
                scene["camera"] = "Medium shot, eye-level tracking shot (matches CENTER PANEL 3/4 Side View of turnaround sheet)"

            if not scene.get("lighting"):
                scene["lighting"] = "Warm cinematic lighting"

            if not scene.get("video_prompt"):
                scene["video_prompt"] = scene.get("action", "Smooth continuous physical motion")

            if not scene.get("audio") or not isinstance(scene.get("audio"), dict):
                scene["audio"] = {
                    "narration": [],
                    "dialogue": [],
                    "sound_effects": ["soft breathing"],
                    "music": {"mood": "sensual", "intensity": "medium"},
                }

            # --------------------------------------------------
            # Auto-populate keyframe-enrichment scene fields if missing / clean placeholders
            # --------------------------------------------------
            for key in ["description", "action", "primary_motion", "pose_lock", "video_prompt"]:
                if scene.get(key):
                    txt = str(scene[key])
                    cleaned = re.sub(r'motion_anal_thrust_action', 'steady deep rhythmic thrusting motion', txt, flags=re.IGNORECASE)
                    cleaned = re.sub(r'motion_[a-zA-Z0-9_]+_action', 'rhythmic thrusting motion', cleaned, flags=re.IGNORECASE)
                    cleaned = re.sub(r'motion_[a-zA-Z0-9_]+', 'rhythmic motion', cleaned, flags=re.IGNORECASE)
                    cleaned = re.sub(r'continuous rhythmic steady deep rhythmic', 'continuous steady deep rhythmic', cleaned, flags=re.IGNORECASE)
                    cleaned = re.sub(r'continuous rhythmic continuous rhythmic', 'continuous rhythmic', cleaned, flags=re.IGNORECASE)
                    scene[key] = cleaned.strip()

            from .continuity import ACTS, EVENTS
            if scene.get("act") not in ACTS:
                raise ValueError(f"Scene {scene.get('id', idx)}: act must be one of {sorted(ACTS)}")

            scene.setdefault("wardrobe_events", [])
            if not isinstance(scene["wardrobe_events"], list):
                raise ValueError(f"Scene {scene.get('id', idx)}: wardrobe_events must be a list")
            for ev in scene["wardrobe_events"]:
                if (
                    not isinstance(ev, dict)
                    or ev.get("event") not in EVENTS
                    or ev.get("character_id") not in character_ids
                ):
                    raise ValueError(f"Scene {scene.get('id', idx)}: bad wardrobe event {ev}")

            # Safe non-overwriting defaults if fields are missing
            if not scene.get("clothing_state"):
                scene["clothing_state"] = "Canonical outfit from character reference"

            if not scene.get("pose_type"):
                scene["pose_type"] = (scene.get("action") or scene.get("description") or "Standing upright")

            if not scene.get("penetration_state"):
                scene["penetration_state"] = "None"

            if not scene.get("genital_alignment"):
                scene["genital_alignment"] = "None"

            if not scene.get("primary_motion"):
                scene["primary_motion"] = (scene.get("video_prompt") or scene.get("action") or "")

            if not scene.get("pose_lock"):
                scene["pose_lock"] = scene.get("pose_type", scene.get("action", ""))

            # Sanitize hand positions to avoid boilerplate contradictions
            if not scene.get("hand_positions"):
                scene["hand_positions"] = "Hands anchored naturally on body or mattress as described in action"
            else:
                hp = str(scene["hand_positions"])
                if "both hands gripping" in hp.lower() and "left hand on mattress" in hp.lower():
                    scene["hand_positions"] = "Man: both hands firmly gripping her hips. Woman: both hands planted on the mattress."

            if not scene.get("contact_zone"):
                scene["contact_zone"] = "Continuous solid skin-to-skin contact between bodies, no dark voids"

            if not scene.get("motion_intensity"):
                scene["motion_intensity"] = "medium"

            if not scene.get("expression"):
                scene["expression"] = "Intense, emotionally grounded facial expression"

            # Ensure structured video prompt (Wan 2.2 5-part format)
            vp = scene.get("video_prompt", "")
            if not ("Exact contact:" in vp and "Hands:" in vp and "Camera:" in vp):
                act_text = scene.get("primary_motion") or scene.get("action") or vp
                contact_text = scene.get("genital_alignment", "Continuous solid skin contact")
                hands_text = scene.get("hand_positions", "Hands anchored securely")
                pose_text = scene.get("pose_lock", scene.get("pose_type", "Locked body pose"))
                camera_text = scene.get("camera", "Cinematic camera framing")
                scene["video_prompt"] = f"{act_text.rstrip('.')}. Exact contact: {contact_text}. Hands: {hands_text}. Bodies remain locked in {pose_text}. Camera: {camera_text}."

            scene_order = scene.get("order", 1)
            if scene_order >= 2 and not scene.get("previous_scene_context"):
                scenes_list = result.get("scenes", [])
                prev_idx = scene_order - 2
                if 0 <= prev_idx < len(scenes_list):
                    prev_scene = scenes_list[prev_idx]
                    prev_desc = prev_scene.get("description", "")
                    prev_clothing = prev_scene.get("clothing_state", "Fully nude, completely unclothed, bare skin")
                    prev_pose = prev_scene.get("pose_type", prev_scene.get("action", ""))
                    prev_align = prev_scene.get("genital_alignment", "None")
                    scene["previous_scene_context"] = f"In scene {scene_order - 1}, character was in pose ({prev_pose}). Clothing state: {prev_clothing}. Genital alignment: {prev_align}. Context: {prev_desc}"

            for field in self.REQUIRED_SCENE_FIELDS:
                if field not in scene:
                    raise ValueError(f"Scene missing field: {field}")

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

        # --------------------------------------------------
        # Post-validation: Dynamic pacing and audio escalation across scene arc
        # If motion_intensity or music intensity is static/flat across all scenes,
        # escalate them naturally to match the narrative arc.
        # --------------------------------------------------
        scenes = result.get("scenes", [])
        if len(scenes) > 1:
            intensities = [s.get("motion_intensity", "").strip().lower() for s in scenes]
            if len(set(intensities)) <= 1:
                total_s = len(scenes)
                for idx, s in enumerate(scenes):
                    ratio = idx / max(1, total_s - 1)
                    if ratio < 0.25:
                        s["motion_intensity"] = "slow"
                    elif ratio < 0.50:
                        s["motion_intensity"] = "deep and slow"
                    elif ratio < 0.75:
                        s["motion_intensity"] = "rhythmic medium"
                    elif ratio < 0.90:
                        s["motion_intensity"] = "fast"
                    else:
                        s["motion_intensity"] = "final deep hold"

            music_moods = [s.get("audio", {}).get("music", {}).get("mood", "").strip().lower() for s in scenes]
            music_intensities = [s.get("audio", {}).get("music", {}).get("intensity", "").strip().lower() for s in scenes]
            if len(set(music_intensities)) <= 1 or len(set(music_moods)) <= 1:
                total_s = len(scenes)
                for idx, s in enumerate(scenes):
                    ratio = idx / max(1, total_s - 1)
                    m = s.setdefault("audio", {}).setdefault("music", {})
                    if ratio < 0.25:
                        m["mood"] = "sensual"
                        m["intensity"] = "low"
                    elif ratio < 0.50:
                        m["mood"] = "sensual"
                        m["intensity"] = "medium"
                    elif ratio < 0.75:
                        m["mood"] = "rhythmic passionate"
                        m["intensity"] = "medium"
                    elif ratio < 0.90:
                        m["mood"] = "intense passionate"
                        m["intensity"] = "high"
                    else:
                        m["mood"] = "climactic passionate"
                        m["intensity"] = "very high"

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