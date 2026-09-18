import json
import logging
import os
import shutil

from .composition_generator import CompositionGenerator
from .config import OUTPUT_DIR
from .keyframe_generator import KeyframeGenerator
from .video_generator import VideoGenerator
from .video_utils import extract_last_frame

logger = logging.getLogger(__name__)


class SceneManager:

    def __init__(self):
        self.composition_generator = (
            CompositionGenerator()
        )
        self.composition_generator = CompositionGenerator()
        self.keyframe_generator = KeyframeGenerator()
        self.video_generator = VideoGenerator()

        self.keyframe_generator = (
            KeyframeGenerator()
        )
    @staticmethod
    def validate_scene(
        scene,
        story_plan,
        references,
        previous_scenes=None,
    ):
        """
        Validate a scene before generation.

        self.video_generator = (
            VideoGenerator()
        Checks:
        - STORY → SCENE: story segment consistency
        - SCENE → CHARACTER: character IDs and references exist
        - SCENE → ENVIRONMENT: location and environment state valid
        - SCENE → CONTINUITY: previous scene state is valid
        - SCENE → PROMPT: required information exists, no contradictions
        """

        scene_id = scene.get("id", "")
        scene_order = scene.get("order", 0)
        character_ids = scene.get("character_ids", [])
        location_id = scene.get("location_id", "")

        missing_errors = []

        # --------------------------------------------------
        # STORY → SCENE validation
        # --------------------------------------------------
        all_scenes = story_plan.get("scenes", [])
        scene_order_valid = any(
            s.get("order") == scene_order for s in all_scenes
        )
        if not scene_order_valid:
            missing_errors.append(
                f"Scene {scene_id}: invalid scene order {scene_order}"
            )

        if not scene.get("description", "").strip():
            missing_errors.append(
                f"Scene {scene_id}: description is empty"
            )

        if not scene.get("action", "").strip():
            missing_errors.append(
                f"Scene {scene_id}: action is empty"
            )

        # --------------------------------------------------
        # SCENE → CHARACTER validation
        # --------------------------------------------------
        for char_id in character_ids:
            if char_id not in references:
                missing_errors.append(
                    f"Scene {scene_id}: no reference for character {char_id}"
                )

        character_map = {
            c["id"]: c for c in story_plan.get("characters", [])
        }
        for char_id in character_ids:
            if char_id not in character_map:
                missing_errors.append(
                    f"Scene {scene_id}: character {char_id} not in story plan"
                )

        # --------------------------------------------------
        # SCENE → ENVIRONMENT validation
        # --------------------------------------------------
        location_map = {
            l["id"]: l for l in story_plan.get("locations", [])
        }
        if location_id and location_id not in location_map:
            missing_errors.append(
                f"Scene {scene_id}: location {location_id} not in story plan"
            )

        # --------------------------------------------------
        # SCENE → CONTINUITY validation
        # --------------------------------------------------
        if previous_scenes and scene_order > 1:
            if isinstance(previous_scenes, list):
                prev_scene = previous_scenes[-1]
                prev_continuity = prev_scene.get("continuity", {})
                for char_id in character_ids:
                    if char_id in prev_continuity:
                        char_cont = prev_continuity.get(char_id, {})
                        if not char_cont.get("clothing") and not char_cont.get("pose"):
                            missing_errors.append(
                                f"Scene {scene_id}: previous scene {prev_scene.get('id')} "
                                f"missing continuity for character {char_id}"
                            )
            elif isinstance(previous_scenes, dict):
                for char_id in character_ids:
                    if char_id in previous_scenes:
                        char_cont = previous_scenes.get(char_id, {})
                        if not isinstance(char_cont, dict):
                            missing_errors.append(
                                f"Scene {scene_id}: invalid continuity state for character {char_id}"
                            )

        # --------------------------------------------------
        # SCENE → PROMPT validation
        # --------------------------------------------------
        if not scene.get("camera", "").strip():
            missing_errors.append(
                f"Scene {scene_id}: camera description is empty"
            )

        if not scene.get("lighting", "").strip():
            missing_errors.append(
                f"Scene {scene_id}: lighting description is empty"
            )

        if missing_errors:
            return False, missing_errors

        return True, []

    @staticmethod
    def build_motion_prompt(scene):
    def build_motion_prompt(scene, scene_state=None):
        description = scene.get("description", "")
        action = scene.get("action", "")
        camera = scene.get("camera", "")
        lighting = scene.get("lighting", "")

        # Detect or force sexual continuity
        sexual_keywords = [
            "sex", "fuck", "penetrat", "oral", "blowjob", "cunnilingus",
            "handjob", "masturbat", "nude", "naked", "breast", "pussy",
            "cock", "penis", "vagina", "orgasm", "cum", "thrust", "moan"
        ]
        is_sexual = any(kw in (description + action).lower() for kw in sexual_keywords)

        base = f"""
Animate this established cinematic scene naturally.
Scene:
Animate this input keyframe with smooth, continuous, realistic physical motion.

Scene description:
{description}
Action:

Action to animate:
{action}

Camera movement:
{camera}

Lighting:
{lighting}

The input keyframe is the visual authority for:
- character identity
- character clothing (or lack of clothing)
- environment
- architecture
- spatial layout
- lighting
- weather
- time of day
- body positions and sexual contact if present
STRICT CONTINUITY AUTHORITY:
- The input keyframe is the absolute visual authority for character faces, identity, clothing (or lack of clothing), room layout, environment, and body positions.
- Do NOT redesign or morph character faces during animation.
- Do NOT change character clothing or room layout.
- Introduce natural, fluid, physically believable motion only.
"""

Introduce realistic motion only.
        if scene_state:
            base += f"""
Environment continuity:
- Location: {scene_state.get('location', 'Not specified')}
- Time of day: {scene_state.get('time_of_day', 'Not specified')}
- Weather: {scene_state.get('weather', 'Not specified')}
"""

        if is_sexual:
            base += """
This is an explicit adult sexual scene. 
Animate the exact sexual action visible in the keyframe with realistic, continuous motion:
- Clear thrusting / penetration / oral / stroking motion
- Natural body movement, hip motion, and muscle tension
- Realistic movement of breasts, genitals, and any remaining clothing
- Visible wetness and skin contact
- Facial expressions of pleasure
Do not stop the sexual activity.
Do not add clothing.
Do not fade to black.
Do not change the sexual act into something non-sexual.
Keep the motion physically believable and continuous for the full duration.
"""
        else:
            base += """
Natural human movement.
Natural clothing movement.
Physically believable environmental motion.
Subtle atmospheric motion.
Subtle atmospheric environmental motion.
Cinematic camera movement.
"""

        base += """
Do not redesign the characters.
Do not change their faces.
Do not change their clothing (or lack of it).
Do not change the location.
Do not introduce additional characters.
Do not create a different scene.
Do not create a collage or split screen.
"""
        return base
    
#     @staticmethod
#     def build_motion_prompt(scene):
#         return f"""
# Animate this established cinematic scene naturally.

# Scene:
# {scene.get("description", "")}

# Action:
# {scene.get("action", "")}

# Camera movement:
# {scene.get("camera", "")}

# Lighting:
# {scene.get("lighting", "")}

# The input keyframe is the visual authority for:
# - character identity
# - character clothing
# - environment
# - architecture
# - spatial layout
# - lighting
# - weather
# - time of day

# Introduce realistic motion only.

# Natural human movement.
# Natural clothing movement.
# Physically believable rain.
# Natural environmental motion.
# Subtle atmospheric motion.
# Cinematic camera movement.

# Do not redesign the characters.
# Do not change their faces.
# Do not change their clothing.
# Do not change the location.
# Do not introduce additional characters.
# Do not create a different scene.
# Do not create a collage or split screen.
# """

    @staticmethod
    def build_scene_state(
        scene,
        previous_state=None,
        character_states=None,
    ):
        """
        Build explicit continuity state for the current scene.

        State is intentionally derived from the story plan and
        previous scene state. It does not invent persistent objects.
        """
        previous_state = previous_state or {}
        character_states = character_states or {}

        state = {
            "scene_id": scene.get("id"),
            "order": scene.get("order"),
            "location": scene.get(
                "location_id"
            ),
            "characters": list(
                scene.get(
                    "character_ids",
                    []
                )
            ),
            "weather": scene.get(
                "weather"
            ),
            "lighting": scene.get(
                "lighting"
            ),
            "objects": dict(
                previous_state.get(
                    "objects",
                    {}
                )
            ),
            "notes": list(
                previous_state.get(
                    "notes",
                    []
                )
            ),
            "location": scene.get("location_id"),
            "characters": list(scene.get("character_ids", [])),
            "weather": scene.get("weather"),
            "lighting": scene.get("lighting"),
            "objects": dict(previous_state.get("objects", {})),
            "notes": list(previous_state.get("notes", [])),
        }

        # Optional continuity section generated by the planner.
        continuity = scene.get(
            "continuity"
        state["character_current_states"] = dict(
            previous_state.get("character_current_states", {})
        )

        continuity = scene.get("continuity")
        if isinstance(continuity, dict):
            for char_id in continuity:
                if char_id in scene.get("character_ids", []):
                    char_continuity = continuity.get(char_id, {})
                    if isinstance(char_continuity, dict):
                        state["character_current_states"][char_id] = {
                            "current_clothing_state": char_continuity.get("clothing", ""),
                            "current_pose": char_continuity.get("pose", ""),
                        }

            persistent_objects = continuity.get(
                "persistent_objects",
                []
            )
        for char_id, cs in character_states.items():
            if char_id in state["character_current_states"]:
                state["character_current_states"][char_id].update(cs)

            if isinstance(
                persistent_objects,
                list,
            ):
                for obj in persistent_objects:
        state["time_of_day"] = previous_state.get("time_of_day")
        state["weather"] = scene.get("weather") or previous_state.get("weather")
        state["lighting"] = scene.get("lighting") or previous_state.get("lighting")
        state["lighting_quality"] = previous_state.get("lighting_quality")
        state["important_objects"] = dict(previous_state.get("important_objects", {}))
        state["spatial_layout"] = previous_state.get("spatial_layout")
        state["background"] = previous_state.get("background")

                    if not isinstance(
                        obj,
                        dict,
                    ):
                        continue

                    obj_id = obj.get("id")

                    if obj_id:
                        state["objects"][obj_id] = obj

            continuity_notes = continuity.get(
                "notes",
                []
            )

            if isinstance(
                continuity_notes,
                list,
            ):
                state["notes"].extend(
                    continuity_notes
                )

            elif continuity_notes:
                state["notes"].append(
                    continuity_notes
                )

        return state

    def generate_keyframes(
        self,
        job_id,
        story_plan,
        references,
        start_scene=1,
        end_scene=None,
    ):
        job_dir = OUTPUT_DIR / job_id

        scenes = story_plan.get(
            "scenes",
            []
        )

        scenes = story_plan.get("scenes", [])
        if not scenes:
            raise RuntimeError(
                "Story plan contains no scenes."
            )
            raise RuntimeError("Story plan contains no scenes.")

        if end_scene is None:
            end_scene = len(scenes)

        results = []

        previous_last_frame = None
        previous_scene_state = None
        previous_character_states = {}

        character_map = {
            character["id"]: character
            for character in story_plan.get(
                "characters",
                []
            )
            for character in story_plan.get("characters", [])
        }

        for scene in scenes:

            order = int(
                scene["order"]
            )

            order = int(scene["order"])
            if order < start_scene:
                continue

            if order > end_scene:
                break

            scene_id = scene["id"]
            scene_id = scene.get("id", "")
            scene_character_ids = scene.get("character_ids", [])
            location_id = scene.get("location_id", "")

            # --------------------------------------------------
            # Persistent scene state
            # --------------------------------------------------
            # Pre-generation validation
            is_valid, validation_errors = self.validate_scene(
                scene,
                story_plan,
                references,
                previous_scene_state.get("character_current_states", {}) if previous_scene_state else {},
            )

            scene_state = (
                self.build_scene_state(
                    scene,
                    previous_scene_state,
            if not is_valid:
                raise RuntimeError(
                    f"Scene {scene_id} validation failed:\n"
                    + "\n".join(f"  - {err}" for err in validation_errors)
                )
            )

            scene_dir = (
                job_dir
                / "scenes"
                / scene_id
            logger.info(f"[SCENE] Starting scene {order}: {scene_id}")
            logger.info(f"[STORY] story_id: {job_id}")
            logger.info(f"[SCENE] scene_id: {scene_id} scene_order: {order}")
            logger.info(f"[CHARACTERS] character_ids: {scene_character_ids}")
            logger.info(f"[ENVIRONMENT] location: {location_id}")
            logger.info(
                f"[CONTINUITY] previous_scene_id: {previous_scene_state.get('scene_id', 'N/A') if previous_scene_state else 'N/A'}"
            )

            scene_dir.mkdir(
                parents=True,
                exist_ok=True,
            logger.info(
                f"[CONTINUITY] inherited_character_state: {previous_character_states}"
            )

            composition_path = (
                scene_dir
                / "composition.png"
            logger.info(
                f"[CONTINUITY] inherited_environment_state: time_of_day={previous_scene_state.get('time_of_day', 'N/A') if previous_scene_state else 'N/A'}"
            )

            keyframe_path = (
                scene_dir
                / "keyframe.png"
            scene_state = self.build_scene_state(
                scene,
                previous_scene_state,
                previous_character_states,
            )

            video_path = (
                scene_dir
                / "video.mp4"
            )
            scene_dir = job_dir / "scenes" / scene_id
            scene_dir.mkdir(parents=True, exist_ok=True)

            last_frame_path = (
                scene_dir
                / "last_frame.png"
            )
            composition_path = scene_dir / "composition.png"
            keyframe_path = scene_dir / "keyframe.png"
            video_path = scene_dir / "video.mp4"
            last_frame_path = scene_dir / "last_frame.png"

            # --------------------------------------------------
            # Character references
            # --------------------------------------------------

            scene_character_ids = scene.get(
                "character_ids",
                []
            )

            character_refs = []
            scene_characters = []

            for char_id in scene_character_ids:

                reference = references.get(
                    char_id
                )

                reference = references.get(char_id)
                if not reference:
                    raise RuntimeError(
                        f"No character reference found "
                        f"for '{char_id}' in scene "
                        f"'{scene_id}'."
                        f"No character reference found for '{char_id}' in scene '{scene_id}'."
                    )

                character_refs.append(
                    reference
                )

                character = character_map.get(
                    char_id
                )

                character_refs.append(reference)
                character = character_map.get(char_id)
                if character:
                    scene_characters.append(
                        character
                    )
                    scene_characters.append(character)

            if not character_refs:
                raise RuntimeError(
                    f"Scene '{scene_id}' contains no characters."
                )
                raise RuntimeError(f"Scene '{scene_id}' contains no characters.")

            if len(character_refs) > 3:
                raise RuntimeError(
                    f"Scene '{scene_id}' contains "
                    f"{len(character_refs)} character references. "
                    f"Maximum supported is 3."
                    f"Scene '{scene_id}' contains {len(character_refs)} character references. Maximum supported is 3."
                )

            # --------------------------------------------------
            # Stage 1
            #
            # Scene 1:
            #     Qwen T2I creates initial composition.
            #
            # Scene 2+:
            #     Actual last frame of previous video is reused.
            # --------------------------------------------------
            # Determine adaptive denoise level & setup stage 1
            prev_loc = previous_scene_state.get("location") if previous_scene_state else None
            curr_loc = location_id

            if previous_last_frame is None:

                # Opening scene
                active_denoise = 0.70
                print()
                print("=" * 80)
                print(
                    f"SCENE {order}: "
                    f"GENERATING INITIAL COMPOSITION"
                )
                print(f"SCENE {order}: GENERATING INITIAL COMPOSITION")
                print("=" * 80)

                composition_result = (
                    self.composition_generator.generate(
                        scene=scene,
                        characters=scene_characters,
                        output_path=composition_path,
                        seed=1000003 + order * 7919,
                    )
                composition_result = self.composition_generator.generate(
                    scene=scene,
                    characters=scene_characters,
                    output_path=composition_path,
                    seed=1000003 + order * 7919,
                )

                base_image = (
                    composition_path
                )

                base_source = (
                    "qwen_t2i"
                )

                base_image = composition_path
                base_source = "qwen_t2i"
            else:
                # Subsequent scene
                if prev_loc and curr_loc and prev_loc == curr_loc:
                    # Same location -> preserve room background & layout
                    active_denoise = 0.50
                else:
                    # New location -> allow room transition
                    active_denoise = 0.65

                print()
                print("=" * 80)
                print(
                    f"SCENE {order}: "
                    f"USING PREVIOUS VIDEO LAST FRAME"
                )
                print(f"SCENE {order}: USING PREVIOUS VIDEO LAST FRAME (denoise={active_denoise})")
                print("=" * 80)
                print("Previous last frame:", previous_last_frame)

                print(
                    "Previous last frame:",
                    previous_last_frame,
                )
                shutil.copy2(previous_last_frame, composition_path)
                base_image = composition_path
                base_source = "previous_video_last_frame"
                composition_result = {"prompt_id": None}

                shutil.copy2(
                    previous_last_frame,
                    composition_path,
                )
            # Build explicit Image slot assignment block for Qwen Edit Plus
            image_mapping_lines = [
                "=========================================================",
                "IMAGE SLOT ASSIGNMENT & IDENTITY AUTHORITY",
                "=========================================================",
                "Image 1: Current Base Scene / Previous Video Last Frame.",
                "Use Image 1 as the primary authority for room background, furniture, spatial layout, and lighting.",
            ]

                base_image = (
                    composition_path
            for idx, char_id in enumerate(scene_character_ids, start=1):
                char_obj = character_map.get(char_id, {})
                c_name = char_obj.get("name", char_id)
                c_anchor = char_obj.get("visual_anchor", char_obj.get("appearance", ""))
                slot_num = idx + 1
                image_mapping_lines.append(
                    f"Image {slot_num}: Reference image authority for character '{c_name}' (ID: {char_id})."
                )

                base_source = (
                    "previous_video_last_frame"
                image_mapping_lines.append(
                    f"  -> Copy {c_name}'s face, eyes, hair, facial structure, skin tone, and body build strictly from Image {slot_num}."
                )

                composition_result = {
                    "prompt_id": None
                }
            image_mapping_block = "\n".join(image_mapping_lines)

            # --------------------------------------------------
            # Stage 2: Qwen Image Edit
            # --------------------------------------------------

            # Stage 2: Keyframe Generation
            print()
            print("=" * 80)
            print(
                f"SCENE {order}: "
                f"GENERATING CONTINUITY KEYFRAME"
            )
            print(f"SCENE {order}: GENERATING CONTINUITY KEYFRAME (denoise={active_denoise})")
            print("=" * 80)

            if previous_last_frame:

                continuity_instruction = f"""
This scene continues from the previous video last frame.
You may change pose, clothing state, and sexual position as required by the NEW scene description and action.
Do not simply copy the previous frame.
Advance the sexual action according to the current scene.
This scene continues from the previous video last frame (Image 1).
Preserve the exact room layout, furniture, background architecture, and lighting from Image 1.
Preserve character poses and clothing states from the previous scene unless the NEW scene description explicitly requires a change.
Advance the scene action smoothly while maintaining character identity from the reference images.
"""

            else:

                continuity_instruction = """
This is the opening scene.

Establish the visual world, characters,
environment and cinematic language of the story.
Establish the visual world, characters, environment, and cinematic language of the story.
"""

            keyframe_prompt = f"""
Create ONE SINGLE photorealistic cinematic frame
that represents the exact story moment described below.

The generated image will become the visual starting point
for the next video-generation stage.
{image_mapping_block}

============================================================
=========================================================
STORY MOMENT
============================================================
=========================================================

Scene ID: {scene_id}
Order: {order}

Scene description:
{scene.get("description", "")}

Action happening in this exact moment:
{scene.get("action", "")}

Camera direction:
{scene.get("camera", "")}

Lighting:
{scene.get("lighting", "")}

============================================================
CONTINUITY
============================================================
Time of day:
{scene_state.get("time_of_day", "Not specified")}

=========================================================
CONTINUITY INSTRUCTIONS
=========================================================
{continuity_instruction}

Continuity state:
{json.dumps(
    scene_state,
    ensure_ascii=False,
    indent=2,
)}
Per-Character Continuity State:
{json.dumps(scene_state.get("character_current_states", {}), ensure_ascii=False, indent=2)}

CRITICAL: This scene contains ONLY the characters listed in character_ids.
Never add extra people.
- If continuing from Image 1, preserve the exact room architecture, furniture, and environmental lighting from Image 1.
- Place the characters in the scene and render their faces strictly matching their corresponding reference images (Image 2, Image 3, etc.).
- Do NOT swap character faces between characters.
- Do NOT introduce unscripted people or background faces.

============================================================
CHARACTER IDENTITY & COUNT
STRICT IMAGE RULES
============================================================
Use the supplied character reference images as the
PRIMARY identity authority.

There are EXACTLY {len(character_refs)} characters in this scene.
Do NOT add any extra people.
Do NOT invent additional characters.
Do NOT create background people.
Do NOT create reflections that look like extra people.
Do NOT create a third person.

Preserve the exact identity of every supplied character:
- facial structure
- facial proportions
- age
- hairstyle
- hair color
- eye color
- skin appearance
- body proportions
- clothing (or nudity) as required by the current scene
- distinctive features
- visual anchor

Do NOT redesign the characters.
Do NOT replace the characters.
Do NOT merge characters.
Do NOT create alternate versions of characters.
Do NOT create look-alikes.

============================================================
STORY ACTION
============================================================

The image must visually represent the action described
in the story.

If a character is:

- walking, show them walking
- entering, show them entering
- leaving, show them leaving
- sitting, show them sitting
- standing, show them standing
- looking at someone, show the correct eyeline
- holding something, show the correct object in their hand
- touching something, show the interaction
- reacting emotionally, show the appropriate facial expression
- speaking to another character, show their physical interaction
- running, show believable running posture
- falling, show believable body position
- hiding, show believable concealment
- interacting with the environment, show that interaction



Do not invent unrelated characters.

Do not create events that are not described by the story.

The frame must represent ONE specific moment,
not several moments combined together.

============================================================
EXPLICIT SEXUAL CONTENT (CRITICAL)
============================================================
If the scene description or action contains sexual activity,
you MUST show it clearly and photorealistically:

- Correct sexual position and body contact
- Visible genitals / penetration / oral contact when described
- Realistic wetness and physical interaction
- Correct facial expressions of pleasure

Do NOT:
- Keep clothes on when the action requires nudity
- Soften or censor the sexual act
- Turn the scene non-sexual
- Create a different action than the one written

============================================================
ENVIRONMENT
============================================================

The environment must remain consistent with the story.

Preserve and correctly depict:

- location
- architecture
- furniture
- important props
- weather
- time of day
- atmosphere
- spatial relationships
- background elements
- foreground elements

Objects mentioned by the story should appear when
they are relevant to the current visual moment.

============================================================
CAMERA
============================================================

Follow the requested camera direction:

{scene.get("camera", "")}

Use exactly ONE physical camera.

ONE viewpoint.

ONE shot.

ONE continuous photographic frame.

Do not show alternate camera angles.

Do not show multiple viewpoints.

============================================================
LIGHTING
============================================================

Follow the requested lighting:

{scene.get("lighting", "")}

Lighting must remain physically believable and consistent
with the environment, weather and time of day.

============================================================
CINEMATIC QUALITY
============================================================

Photorealistic live-action feature-film appearance.

Realistic human anatomy.

Realistic facial proportions.

Realistic skin texture.

Realistic hair.

Realistic clothing materials.

Realistic environments.

Natural perspective.

Natural depth of field.

Physically believable lighting.

Natural shadows.

Subtle cinematic color grading.

High-end feature-film cinematography.

The image must look like an actual frame captured
from a professionally produced live-action movie.

============================================================
STRICT IMAGE RULES
============================================================

ONE image.

ONE moment.

ONE continuous frame.

ONE camera.

ONE viewpoint.

No character sheet.

No reference sheet.

No turnaround sheet.

No multiple views.

No alternate poses.

No collage.

No split screen.

No storyboard.

No comic panels.

No multiple panels.

No duplicate characters.

No duplicated faces.

No duplicated bodies.

No mirrored characters.

No double exposure.

No picture-in-picture.

No montage.

No unrelated people.

No text.

No captions.

No subtitles.

No labels.

No watermark.

No logo.

Generate the exact cinematic story moment
while preserving character and scene continuity.
ONE camera viewpoint.
No collage, no split screen, no turnaround sheet, no text, no subtitles.
Generate the exact cinematic story moment while preserving character identity and scene continuity.
"""

            keyframe_result = (
                self.keyframe_generator.generate(
                    scene_image=base_image,
                    character_references=character_refs,
                    prompt=keyframe_prompt,
                    output_path=keyframe_path,
                    seed=2000003 + order * 7919,
                )
            keyframe_result = self.keyframe_generator.generate(
                scene_image=base_image,
                character_references=character_refs,
                prompt=keyframe_prompt,
                output_path=keyframe_path,
                seed=2000003 + order * 7919,
                denoise=active_denoise,
            )

            # --------------------------------------------------
            # Stage 3: Wan 2.2 I2V
            # --------------------------------------------------

            # Stage 3: Wan 2.2 Video Generation
            print()
            print("=" * 80)
            print(
                f"SCENE {order}: "
                f"GENERATING WAN 2.2 VIDEO"
            )
            print(f"SCENE {order}: GENERATING WAN 2.2 VIDEO")
            print("=" * 80)

            motion_prompt = (
                self.build_motion_prompt(
                    scene
                )
            motion_prompt = self.build_motion_prompt(
                scene,
                scene_state=scene_state,
            )

            video_result = (
                self.video_generator.generate(
                    keyframe=keyframe_path,
                    motion_prompt=motion_prompt,
                    output_path=video_path,
                    seed=3000003 + order * 7919,
                )
            video_result = self.video_generator.generate(
                keyframe=keyframe_path,
                motion_prompt=motion_prompt,
                output_path=video_path,
                seed=3000003 + order * 7919,
            )

            # --------------------------------------------------
            # Stage 4: extract ACTUAL final video frame
            # --------------------------------------------------

            # Stage 4: Last frame extraction
            print()
            print("=" * 80)
            print(
                f"SCENE {order}: "
                f"EXTRACTING LAST FRAME"
            )
            print(f"SCENE {order}: EXTRACTING LAST FRAME")
            print("=" * 80)

            extract_last_frame(
                video_path,
                last_frame_path,
            )
            extract_last_frame(video_path, last_frame_path)

            # --------------------------------------------------
            # Metadata
            # --------------------------------------------------

            metadata = {
                "scene_id": scene_id,
                "order": order,
                "character_ids": scene_character_ids,
                "character_references": character_refs,
                "composition": str(composition_path),
                "composition_source": base_source,
                "keyframe": str(keyframe_path),
                "video": str(video_path),
                "last_frame": str(last_frame_path),
                "previous_last_frame": str(previous_last_frame) if previous_last_frame else None,
                "composition_prompt_id": composition_result.get("prompt_id"),
                "keyframe_prompt_id": keyframe_result.get("prompt_id"),
                "video_prompt_id": video_result.get("prompt_id"),
                "video_generation_time_seconds": video_result.get("generation_time_seconds"),
                "scene_state": scene_state,
                "active_denoise": active_denoise,
                "story_id": job_id,
                "scene_segment": scene.get("description", "")[:80] if scene.get("description") else "",
                "characters_used": scene_character_ids,
                "location": location_id,
                "action": scene.get("action", ""),
                "generated_prompt": keyframe_prompt[:200] if keyframe_prompt else "",
                "generated_video": str(video_path),
                "generation_status": "completed",
                "validation_status": "pending",
            }

                "character_ids": (
                    scene_character_ids
                ),
            gen_valid, gen_errors = self.validate_generated_scene(metadata)
            if not gen_valid:
                raise RuntimeError(
                    f"Scene {scene_id} post-generation validation failed:\n"
                    + "\n".join(f"  - {err}" for err in gen_errors)
                )

                "character_references": (
                    character_refs
                ),

                "composition": (
                    str(
                        composition_path
                    )
                ),

                "composition_source": (
                    base_source
                ),

                "keyframe": (
                    str(
                        keyframe_path
                    )
                ),

                "video": (
                    str(
                        video_path
                    )
                ),

                "last_frame": (
                    str(
                        last_frame_path
                    )
                ),

                "previous_last_frame": (
                    str(
                        previous_last_frame
                    )
                    if previous_last_frame
                    else None
                ),

                "composition_prompt_id": (
                    composition_result.get(
                        "prompt_id"
                    )
                ),

                "keyframe_prompt_id": (
                    keyframe_result.get(
                        "prompt_id"
                    )
                ),

                "video_prompt_id": (
                    video_result.get(
                        "prompt_id"
                    )
                ),

                "video_generation_time_seconds": (
                    video_result.get(
                        "generation_time_seconds"
                    )
                ),

                "scene_state": (
                    scene_state
                ),
            }

            with open(
                scene_dir / "metadata.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    metadata,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            results.append(
                metadata
            )
            results.append(metadata)

            # --------------------------------------------------
            # Current scene becomes continuity state
            # for the next scene.
            # --------------------------------------------------
            previous_last_frame = last_frame_path
            previous_scene_state = scene_state
            previous_character_states = scene_state.get("character_current_states", {})

            previous_last_frame = (
                last_frame_path
        return results

    @staticmethod
    def validate_generated_scene(metadata):
        errors = []
        scene_id = metadata.get("scene_id", "")
        character_refs = metadata.get("character_references", [])
        scene_state = metadata.get("scene_state", {})
        video_path = metadata.get("video", "")

        for ref_path in character_refs:
            if not ref_path or not ref_path.strip():
                errors.append(f"Scene {scene_id}: empty character reference path")

        expected_char_count = len(metadata.get("character_ids", []))
        actual_char_count = len(character_refs)
        if expected_char_count != actual_char_count:
            errors.append(
                f"Scene {scene_id}: character count mismatch: expected {expected_char_count}, got {actual_char_count}"
            )

            previous_scene_state = (
                scene_state
        if not scene_state.get("time_of_day") and not scene_state.get("lighting"):
            errors.append(
                f"Scene {scene_id}: missing environment state (time_of_day/lighting)"
            )

        return results
        if video_path and not os.path.exists(video_path):
            errors.append(f"Scene {scene_id}: generated video file not found: {video_path}")

        last_frame = metadata.get("last_frame", "")
        if last_frame and not os.path.exists(last_frame):
            errors.append(f"Scene {scene_id}: last frame file not found: {last_frame}")

        if errors:
            return False, errors

        return True, []