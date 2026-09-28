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

    def __init__(
        self,
        composition_generator=None,
        keyframe_generator=None,
        video_generator=None,
    ):
        self.composition_generator = composition_generator or CompositionGenerator()
        self.keyframe_generator = keyframe_generator or KeyframeGenerator()
        self.video_generator = video_generator or VideoGenerator()

    @staticmethod
    def validate_scene(
        scene,
        story_plan,
        references,
        previous_scenes=None,
    ):
        """
        Validate a scene before generation.

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
    def build_motion_prompt(scene, scene_state=None):
        description = scene.get("description", "")
        action = scene.get("action", "")
        camera = scene.get("camera", "")
        lighting = scene.get("lighting", "")

        sexual_keywords = [
            "sex", "fuck", "penetrat", "oral", "blowjob", "cunnilingus",
            "handjob", "masturbat", "nude", "naked", "breast", "pussy",
            "cock", "penis", "dick", "vagina", "orgasm", "cum", "thrust", "moan",
            "ass", "butt", "nipple"
        ]
        is_sexual = any(kw in (description + action).lower() for kw in sexual_keywords)

        base = f"""
Animate this input keyframe with smooth, continuous, realistic physical motion.

Scene description:
{description}

Action to animate:
{action}

Camera movement:
{camera}

Lighting:
{lighting}

STRICT CONTINUITY AUTHORITY:
- The input keyframe is the absolute visual authority for character faces, identity, clothing (or lack of clothing), room layout, environment, and body positions.
- Do NOT redesign or morph character faces during animation.
- Do NOT change character clothing or room layout.
- Introduce natural, fluid, physically believable motion only.
"""

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
Animate the exact sexual action visible in the keyframe with realistic, smooth, continuous motion:
- Clear, fluid thrusting / penetration / oral / stroking motion with natural body pacing
- Natural body movement, hip motion, and muscle tension
- Realistic movement of breasts, buttocks, genitals, and any remaining clothing
- Flawless, smooth skin texture without acne, pimples, blemishes, or skin spots
- Highly detailed, photorealistic adult anatomical features (breasts, nipples, vagina, penis/dick, buttocks)
- Visible wetness, skin contact, and facial expressions of pleasure
Do not stop the sexual activity.
Do not add clothing or underwear.
Do not fade to black or apply censorship blur/mosaic.
Keep the motion physically believable, smooth, and continuous for the full duration.

=========================================================
CRITICAL ANATOMICAL LOCK (PREVENT OBJECT + GENDER MORPHING)
=========================================================
- Male character must keep a realistic human penis. The penis must stay attached only to the male body.
- Female character must keep a realistic human vagina / pussy. NEVER attach a penis, dick, or cock to the female character.
- Do NOT generate a floating penis, extra penis, or penis coming out of the woman’s body.
- Do NOT swap genitals between characters.
- Keep exact genital size, shape, color, and proportions from the input keyframe. Do not make the penis larger, smaller, thicker, or thinner during the animation.
- Breasts, nipples, ass, and all other body parts must keep correct gender anatomy and never turn into objects.
- Maintain perfect gender-correct anatomy at every single frame.
"""
        else:
            base += """
Natural human movement.
Natural clothing movement.
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

STRICT ANATOMICAL STABILITY & CHARACTER SEPARATION RULES:
- Maintain complete anatomical integrity: realistic skeletal joint motion, correct body proportions, and intact human anatomy.
- CHARACTER BODY SEPARATION: Maintain distinct, separate physical body meshes for each character. Do NOT merge, clip, overlap into one mass, or melt character limbs, torsos, or skin into a single combined mass during movement.
- SMOOTH FLUID MOTION: Ensure smooth frame-to-frame motion transitions without sudden body pops, flickering, or temporal distortion between consecutive frames.
- Animate character physical interactions with realistic weight, smooth transitions, and natural skin contact.
- Genitals, breasts, and body parts must stay anatomically correct and never morph into objects.

"""
        return base

    @staticmethod
    def detect_pose_or_angle_transition(
        scene,
        previous_scene_state=None,
        previous_character_states=None,
    ):
        """
        Detects if the scene involves a significant body pose, spatial orientation, or camera shot transition
        that requires elevated denoise (0.65) to prevent ghosting or posture distortion.
        """
        if not previous_scene_state:
            return False

        action_text = f"{scene.get('description', '')} {scene.get('action', '')} {scene.get('camera', '')}".lower()

        # Keywords indicating posture / physical orientation changes
        pose_keywords = [
            "roll", "crawl", "turn", "straddle", "lie down", "lying", "climb", "rotate",
            "flip", "mount", "kneel", "kneeling", "on back", "behind", "shift", "reposition",
            "spin", "swap", "over", "under", "across", "sideways", "standing up", "sit up",
            "lay down", "doggy", "missionary", "cowgirl", "69", "spooning", "cradled", "bent over"
        ]

        # Camera angle change keywords that alter perspective significantly
        camera_keywords = [
            "overhead", "close-up", "wide shot", "side view", "reverse angle", "angle shift",
            "high angle", "low angle", "pan around", "zoom in", "tracking shot"
        ]

        if any(kw in action_text for kw in pose_keywords + camera_keywords):
            return True

        # Check explicit continuity pose diff if available
        current_continuity = scene.get("continuity", {})
        if isinstance(current_continuity, dict) and previous_character_states:
            for char_id, curr_char_data in current_continuity.items():
                if isinstance(curr_char_data, dict):
                    curr_pose = curr_char_data.get("pose", "").lower()
                    prev_pose = previous_character_states.get(char_id, {}).get("current_pose", "").lower()
                    if curr_pose and prev_pose and curr_pose != prev_pose:
                        return True

        return False

    @staticmethod
    def build_scene_state(
        scene,
        previous_state=None,
        character_states=None,
    ):
        """
        Build explicit continuity state for the current scene.
        """
        previous_state = previous_state or {}
        character_states = character_states or {}

        state = {
            "scene_id": scene.get("id"),
            "order": scene.get("order"),
            "location": scene.get("location_id"),
            "characters": list(scene.get("character_ids", [])),
            "weather": scene.get("weather"),
            "lighting": scene.get("lighting"),
            "objects": dict(previous_state.get("objects", {})),
            "notes": list(previous_state.get("notes", [])),
        }

        state["character_current_states"] = dict(
            previous_state.get("character_current_states", {})
        )

        for char_id in scene.get("character_ids", []):
            if char_id not in state["character_current_states"]:
                state["character_current_states"][char_id] = {}
            
            curr_cstate = state["character_current_states"][char_id]
            if not curr_cstate.get("current_pose"):
                text_context = f"{scene.get('description', '')} {scene.get('action', '')}".lower()
                if any(w in text_context for w in ["lying", "lies back", "shudders on the bed", "reclining"]):
                    curr_cstate["current_pose"] = "lying down on bed"
                elif any(w in text_context for w in ["sitting", "sits"]):
                    curr_cstate["current_pose"] = "sitting on edge of bed"
                elif any(w in text_context for w in ["standing", "stands"]):
                    curr_cstate["current_pose"] = "standing"

        for char_id, cs in character_states.items():
            if char_id in state["character_current_states"]:
                state["character_current_states"][char_id].update(cs)

        state["time_of_day"] = previous_state.get("time_of_day")
        state["weather"] = scene.get("weather") or previous_state.get("weather")
        state["master_lighting"] = previous_state.get("master_lighting") or scene.get("lighting", "")
        state["lighting"] = scene.get("lighting") or previous_state.get("lighting")
        state["lighting_quality"] = previous_state.get("lighting_quality")
        state["important_objects"] = dict(previous_state.get("important_objects", {}))
        state["spatial_layout"] = previous_state.get("spatial_layout")
        state["background"] = previous_state.get("background")

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

        scenes = story_plan.get("scenes", [])
        if not scenes:
            raise RuntimeError("Story plan contains no scenes.")

        if end_scene is None:
            end_scene = len(scenes)

        results = []
        previous_last_frame = None
        previous_scene_state = None
        previous_character_states = {}

        character_map = {
            character["id"]: character
            for character in story_plan.get("characters", [])
        }

        for scene in scenes:
            order = int(scene["order"])
            if order < start_scene:
                continue
            if order > end_scene:
                break

            scene_id = scene.get("id", "")
            scene_character_ids = scene.get("character_ids", [])
            location_id = scene.get("location_id", "")

            # Pre-generation validation
            is_valid, validation_errors = self.validate_scene(
                scene,
                story_plan,
                references,
                previous_scene_state.get("character_current_states", {}) if previous_scene_state else {},
            )

            if not is_valid:
                raise RuntimeError(
                    f"Scene {scene_id} validation failed:\n"
                    + "\n".join(f"  - {err}" for err in validation_errors)
                )

            logger.info(f"[SCENE] Starting scene {order}: {scene_id}")
            logger.info(f"[STORY] story_id: {job_id}")
            logger.info(f"[SCENE] scene_id: {scene_id} scene_order: {order}")
            logger.info(f"[CHARACTERS] character_ids: {scene_character_ids}")
            logger.info(f"[ENVIRONMENT] location: {location_id}")
            logger.info(
                f"[CONTINUITY] previous_scene_id: {previous_scene_state.get('scene_id', 'N/A') if previous_scene_state else 'N/A'}"
            )
            logger.info(
                f"[CONTINUITY] inherited_character_state: {previous_character_states}"
            )
            logger.info(
                f"[CONTINUITY] inherited_environment_state: time_of_day={previous_scene_state.get('time_of_day', 'N/A') if previous_scene_state else 'N/A'}"
            )

            scene_state = self.build_scene_state(
                scene,
                previous_scene_state,
                previous_character_states,
            )

            scene_dir = job_dir / "scenes" / scene_id
            scene_dir.mkdir(parents=True, exist_ok=True)

            composition_path = scene_dir / "composition.png"
            keyframe_path = scene_dir / "keyframe.png"
            video_path = scene_dir / "video.mp4"
            last_frame_path = scene_dir / "last_frame.png"

            character_refs = []
            scene_characters = []
            for char_id in scene_character_ids:
                reference = references.get(char_id)
                if not reference:
                    raise RuntimeError(
                        f"No character reference found for '{char_id}' in scene '{scene_id}'."
                    )
                character_refs.append(reference)
                character = character_map.get(char_id)
                if character:
                    scene_characters.append(character)

            if not character_refs:
                raise RuntimeError(f"Scene '{scene_id}' contains no characters.")

            if len(character_refs) > 3:
                raise RuntimeError(
                    f"Scene '{scene_id}' contains {len(character_refs)} character references. Maximum supported is 3."
                )

            # --------------------------------------------------
            # Check if scene is sexual or nude
            # --------------------------------------------------
            desc_text = f"{scene.get('description', '')} {scene.get('action', '')}".lower()
            char_current = scene_state.get("character_current_states", {})
            is_nude_scene = False
            for char_id, cstate in char_current.items():
                if isinstance(cstate, dict) and "nude" in str(cstate.get("current_clothing_state", "")).lower():
                    is_nude_scene = True
            if any(w in desc_text for w in [
                "nude", "naked", "unclothed", "bare breast", "undress", "strips",
                "penis", "pussy", "cock", "dick", "vagina", "clitoris", "cum", "ejaculat",
                "sex", "fuck", "penetrat", "doggystyle", "cowgirl", "blowjob", "handjob",
                "oral sex", "grind", "thrust", "intercourse"
            ]):
                is_nude_scene = True

            # --------------------------------------------------
            # Collect genital references ONLY if scene is sexual or nude
            # --------------------------------------------------
            genital_refs = []
            if is_nude_scene:
                for char_id in scene_character_ids:
                    char_obj = character_map.get(char_id, {})
                    appearance = char_obj.get("appearance", char_obj.get("visual_anchor", ""))
                    gender_str = char_obj.get("gender", "")
                    if not gender_str:
                        anchor_lower = (appearance + " " + char_obj.get("name", "")).lower()
                        gender_str = "female" if any(w in anchor_lower for w in ["female", "woman", "girl", "lady"]) else "male"

                    refs = self.get_or_generate_genital_refs(
                        character_appearance=appearance,
                        gender=gender_str,
                        character_id=char_id,
                        output_dir=job_dir / "genital_refs",
                    )
                    genital_refs.extend(refs)

                genital_refs = genital_refs[:2]
            else:
                logger.info(f"[GENITAL REF] Scene {scene_id} is not sexual/nude -> skipping genital references.")

            # Determine adaptive denoise level & setup stage 1
            prev_loc = previous_scene_state.get("location") if previous_scene_state else None
            curr_loc = location_id

            has_pose_transition = self.detect_pose_or_angle_transition(
                scene=scene,
                previous_scene_state=previous_scene_state,
                previous_character_states=previous_character_states,
            )

            if previous_last_frame is None:
                # Opening scene
                active_denoise = 0.70
                print()
                print("=" * 80)
                print(f"SCENE {order}: GENERATING INITIAL COMPOSITION")
                print("=" * 80)

                composition_result = self.composition_generator.generate(
                    scene=scene,
                    characters=scene_characters,
                    output_path=composition_path,
                    seed=1000003 + order * 7919,
                )
                base_image = composition_path
                base_source = "qwen_t2i"
            else:
                # Subsequent scene
                cam_text = f"{scene.get('camera', '')} {scene.get('description', '')}".lower()
                is_close_up = any(w in cam_text for w in ["close-up", "close up", "extreme close-up", "macro", "tight shot"])
                is_wide_or_full = any(w in cam_text for w in ["wide shot", "wide angle", "full shot", "medium full shot", "full-body frame", "full body shot"])

                # Check if previous scene was a close-up
                prev_cam_text = ""
                if order > 1 and len(scenes) >= (order - 1):
                    prev_scene_obj = scenes[order - 2]
                    prev_cam_text = f"{prev_scene_obj.get('camera', '')} {prev_scene_obj.get('description', '')}".lower()
                prev_was_close_up = any(w in prev_cam_text for w in ["close-up", "close up", "extreme close-up", "macro", "tight shot"])

                if prev_loc and curr_loc and prev_loc == curr_loc:
                    if is_close_up or (is_wide_or_full and prev_was_close_up):
                        active_denoise = 0.85
                        logger.info(
                            f"[DENOISE] Camera cut/framing transition (is_close_up={is_close_up}, prev_was_close_up={prev_was_close_up}) in scene {order} -> boosted denoise to {active_denoise}"
                        )
                    elif has_pose_transition:
                        active_denoise = 0.65
                        logger.info(
                            f"[DENOISE] Same location but pose/angle transition detected in scene {order} -> boosted denoise to {active_denoise}"
                        )
                    else:
                        active_denoise = 0.50
                else:
                    # New location -> allow room transition
                    active_denoise = 0.85 if (is_close_up or is_wide_or_full) else 0.65

                print()
                print("=" * 80)
                print(f"SCENE {order}: USING PREVIOUS VIDEO LAST FRAME (denoise={active_denoise}, pose_transition={has_pose_transition})")
                print("=" * 80)
                print("Previous last frame:", previous_last_frame)

                shutil.copy2(previous_last_frame, composition_path)
                base_image = composition_path
                base_source = "previous_video_last_frame"
                composition_result = {"prompt_id": None}

            # Build explicit Image slot assignment block for Qwen Edit Plus
            image_mapping_lines = [
                "=========================================================",
                "IMAGE SLOT ASSIGNMENT & 3D TURNAROUND ANGLE MATCHING",
                "=========================================================",
                "Image 1: Current Base Scene / Previous Video Last Frame.",
                "Use Image 1 as primary authority for room background, furniture, spatial layout, and lighting.",
            ]

            for idx, char_id in enumerate(scene_character_ids, start=1):
                char_obj = character_map.get(char_id, {})
                c_name = char_obj.get("name", char_id)
                c_anchor = char_obj.get("visual_anchor", char_obj.get("appearance", ""))
                c_gender = "male" if "man" in c_anchor.lower() or "male" in c_anchor.lower() else ("female" if "woman" in c_anchor.lower() or "female" in c_anchor.lower() else "person")
                slot_num = idx + 1
                image_mapping_lines.append(
                    f"Image {slot_num}: 3-Panel Turnaround Model Sheet Grid for character '{c_name}' (ID: {char_id}, Gender: {c_gender.upper()})."
                )
                image_mapping_lines.append(
                    f"  -> PANEL LAYOUT: [Left Panel = Front View] | [Center Panel = 3/4 Side View] | [Right Panel = Rear/Back View]."
                )
                image_mapping_lines.append(
                    f"  -> MANDATORY CAMERA ANGLE MATCHING RULES:"
                )
                image_mapping_lines.append(
                    f"     1. FRONT ANGLE SHOT (facing camera / front view): Copy face, eyes, hair, and clothing strictly from the LEFT PANEL (Front View)."
                )
                image_mapping_lines.append(
                    f"     2. PROFILE / SIDE ANGLE SHOT (side view / 3/4 angle): Copy nose profile, jawline, side body structure, and hair strictly from the CENTER PANEL (3/4 Side View)."
                )
                image_mapping_lines.append(
                    f"     3. REAR / BACK ANGLE SHOT (from behind / doggystyle / back view): Copy hair length, back posture, and rear silhouette strictly from the RIGHT PANEL (Rear/Back View)."
                )
                image_mapping_lines.append(
                    f"  -> CRITICAL: Keep {c_name}'s upper body anatomy and clothing state appropriate for {c_name} ({c_gender}). Do NOT render female sports tops, bras, or lingerie on male characters!"
                )

            image_mapping_block = "\n".join(image_mapping_lines)

            # Check for explicit Nudity directive in scene description or continuity
            is_nude_scene = False
            desc_text = f"{scene.get('description', '')} {scene.get('action', '')}".lower()
            char_current = scene_state.get("character_current_states", {})
            for char_id, cstate in char_current.items():
                if isinstance(cstate, dict) and "nude" in str(cstate.get("current_clothing_state", "")).lower():
                    is_nude_scene = True
            if any(w in desc_text for w in ["nude", "undress", "strips", "naked", "bare breast"]):
                is_nude_scene = True

            nudity_instruction = ""
            if is_nude_scene:
                nudity_instruction = """
=========================================================
EXPLICIT NUDITY & UNCLOTHED DIRECTIVE
=========================================================
- THIS SCENE REQUIRES NUDITY / UNCLOTHED CHARACTERS.
- Fully remove bras, tops, lingerie, panties, or clothing on the nude character(s) as requested in the scene action.
- DO NOT render arbitrary white tops, sports bras, wrist wraps, or unwanted underwear when the character is nude on the bed.
"""

            # Build human count & phantom limb elimination block
            char_names_str = ", ".join([character_map.get(cid, {}).get("name", cid) for cid in scene_character_ids])
            human_count_block = f"""
=========================================================
STRICT HUMAN COUNT & PHANTOM LIMB ELIMINATION
=========================================================
- EXACTLY {len(scene_character_ids)} PERSON(S) IN THIS SCENE: {char_names_str}.
- DO NOT render a 3rd person, phantom extra legs on the bed, extra arms, or duplicate bodies from Image 1.
- Fully overwrite and erase any old character body positions or legs on the bed/furniture from Image 1.
- No duplicate mirror reflection clones or second person standing in mirror background.
"""

            # Build pose transition guidance block if pose shift is detected
            pose_transition_instruction = ""
            if has_pose_transition and previous_last_frame:
                pose_transition_instruction = """
=========================================================
POSE & POSITION TRANSITION INSTRUCTION
=========================================================
- THIS SCENE INVOLVES A CHARACTER POSE OR POSITION TRANSITION.
- Completely redraft character body postures, limbs, and spatial orientations as requested in the new scene description.
- DO NOT ghost or blend the new poses on top of old body shapes from Image 1. Re-render clean, anatomically perfect character bodies in their new positions.
- Retain room architecture, background layout, and lighting from Image 1, but rebuild character body postures cleanly.
"""

            # Stage 2: Keyframe Generation
            print()
            print("=" * 80)
            print(f"SCENE {order}: GENERATING CONTINUITY KEYFRAME (denoise={active_denoise})")
            print("=" * 80)

            if previous_last_frame:
                continuity_instruction = f"""
This scene continues from the previous video last frame (Image 1).
Preserve the exact room layout, furniture, background architecture, and lighting from Image 1.
Preserve character clothing states from the previous scene unless the NEW scene description explicitly requires a change.
Advance the scene action smoothly while maintaining character identity from the reference images.
"""
            else:
                continuity_instruction = """
This is the opening scene.
Establish the visual world, characters, environment, and cinematic language of the story.
"""

            master_lighting = scene_state.get("master_lighting") or scene.get("lighting", "")

            keyframe_prompt = f"""
Create ONE SINGLE photorealistic cinematic frame
that represents the exact story moment described below.

=========================================================
CRITICAL GENITAL & BODY PART IDENTITY LOCK
=========================================================
- Male characters must have a realistic human penis attached only to the male body.
- Female characters must have a realistic human vagina only. NEVER render a penis on a female character.
- Do not attach, float, or transfer a penis onto the woman.
- Do not invent extra genitals.
- Keep exact genital size and proportions consistent with the reference and previous frame.
- Breasts, nipples, ass, hands and all body parts must keep correct human anatomy and never become objects.

{image_mapping_block}

{human_count_block}

=========================================================
STORY MOMENT
=========================================================

Scene ID: {scene_id}
Order: {order}

Scene description:
{scene.get("description", "")}

Action happening in this exact moment:
{scene.get("action", "")}

Camera direction:
{scene.get("camera", "")}

Master Lighting Authority (Room Ambient Glow):
{master_lighting}

Scene Lighting Details:
{scene.get("lighting", "")}

Time of day:
{scene_state.get("time_of_day", "Not specified")}

=========================================================
CONTINUITY INSTRUCTIONS
=========================================================
{continuity_instruction}
{nudity_instruction}
{pose_transition_instruction}
Per-Character Continuity State:
{json.dumps(scene_state.get("character_current_states", {}), ensure_ascii=False, indent=2)}

- CRITICAL LIGHTING LOCK: Preserve the exact master room lighting ({master_lighting}) from Image 1. Maintain consistent color temperature, white balance, ambient exposure, and shadow balance across all scene cuts.
- If continuing from Image 1, preserve the exact room architecture, furniture, and environmental lighting from Image 1.
- CHARACTER IDENTITY & BODY CONSISTENCY: Maintain exact facial features, hair color, skin tone, gender traits, and body proportions for each character based on their respective reference images (Image 2, Image 3, etc.).
- Do NOT swap character faces or body features between characters.
- Do NOT alter character physical builds or introduce unwanted objects/artifacts onto characters.
- Do NOT introduce unscripted extra people, extra limbs, or duplicate bodies.

============================================================
STRICT IMAGE RULES
============================================================
ONE continuous photographic frame.
ONE camera viewpoint.
No collage, no split screen, no turnaround sheet, no text, no subtitles.
Generate the exact cinematic story moment while preserving character identity and scene continuity.
"""

            keyframe_result = self.keyframe_generator.generate(
                scene_image=base_image,
                character_references=character_refs,
                prompt=keyframe_prompt,
                output_path=keyframe_path,
                seed=2000003 + order * 7919,
                denoise=active_denoise,
                genital_references=genital_refs,

            )

            # Stage 3: Wan 2.2 Video Generation
            print()
            print("=" * 80)
            print(f"SCENE {order}: GENERATING WAN 2.2 VIDEO")
            print("=" * 80)

            motion_prompt = self.build_motion_prompt(
                scene,
                scene_state=scene_state,
            )

            video_result = self.video_generator.generate(
                keyframe=keyframe_path,
                motion_prompt=motion_prompt,
                output_path=video_path,
                seed=3000003 + order * 7919,
            )

            # Stage 4: Last frame extraction
            print()
            print("=" * 80)
            print(f"SCENE {order}: EXTRACTING LAST FRAME")
            print("=" * 80)

            extract_last_frame(video_path, last_frame_path)

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

            gen_valid, gen_errors = self.validate_generated_scene(metadata)
            if not gen_valid:
                raise RuntimeError(
                    f"Scene {scene_id} post-generation validation failed:\n"
                    + "\n".join(f"  - {err}" for err in gen_errors)
                )

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

            results.append(metadata)

            previous_last_frame = last_frame_path
            previous_scene_state = scene_state
            previous_character_states = scene_state.get("character_current_states", {})

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

        if not scene_state.get("time_of_day") and not scene_state.get("lighting"):
            errors.append(
                f"Scene {scene_id}: missing environment state (time_of_day/lighting)"
            )

        if video_path and not os.path.exists(video_path):
            errors.append(f"Scene {scene_id}: generated video file not found: {video_path}")

        last_frame = metadata.get("last_frame", "")
        if last_frame and not os.path.exists(last_frame):
            errors.append(f"Scene {scene_id}: last frame file not found: {last_frame}")

        if errors:
            return False, errors

        return True, []


    def get_or_generate_genital_refs(self, character_id, character_appearance, gender, output_dir):
        """
        Generate genital close-up reference once per character based on gender and reuse it.
        Returns a list of paths containing [ref_path] or empty list.
        """
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        is_male = str(gender).lower() == "male"
        ref_filename = f"{character_id}_penis_ref.png" if is_male else f"{character_id}_pussy_ref.png"
        ref_path = output_dir / ref_filename

        # If reference image already exists and is non-empty, return it
        if ref_path.exists() and ref_path.stat().st_size > 0:
            return [str(ref_path)]

        # Prompt definition for multi-angle genital grid generation
        if is_male:
            genital_prompt = (
                f"Photorealistic medical anatomical detail turnaround grid, showing 2 clean distinct views of adult male human penis, "
                f"matching skin tone and body type: {character_appearance}: "
                "Left panel: clear frontal view; Right panel: 3/4 side profile view. "
                "Natural skin texture, detailed veins, defined glans, soft natural lighting, "
                "neutral background, highly detailed anatomical accuracy, sharp focus, no text"
            )
        else:
            genital_prompt = (
                f"Photorealistic medical anatomical detail turnaround grid, showing 2 clean distinct views of adult female human vulva / pussy, "
                f"matching skin tone: {character_appearance}: "
                "Left panel: clear frontal view; Right panel: 3/4 side profile view. "
                "Natural labia shape and color, soft natural lighting, "
                "neutral background, highly detailed anatomical accuracy, sharp focus, no text"
            )

        # Generate the genital reference image via composition generator (Krea T2I)
        try:
            logger.info(f"Generating genital reference for character '{character_id}' ({gender})...")
            gen_seed = (abs(hash(character_id)) % 8000000) + 100000
            self.composition_generator.generate_simple(
                prompt=genital_prompt,
                output_path=ref_path,
                seed=gen_seed,
            )
        except Exception as e:
            logger.error(f"Failed to generate genital reference for '{character_id}': {e}")
            return []

        if not ref_path.exists() or ref_path.stat().st_size == 0:
            logger.warning(
                f"Genital reference for '{character_id}' ({gender}) failed to save at {ref_path}."
            )
            return []

        return [str(ref_path)]