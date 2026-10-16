import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

from .composition_generator import CompositionGenerator
from .config import OUTPUT_DIR
from .keyframe_generator import KeyframeGenerator
from .story_schema import motion_to_prompt
from .video_generator import VideoGenerator
from .video_utils import extract_last_frame
from .continuity import (
    ContinuityTracker,
    per_character_clothing_block,
    anatomy_lock,
    dynamic_negative,
    needs_anatomy_pass,
)

logger = logging.getLogger(__name__)


def plan_keyframe(cur: Dict[str, Any], prev: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Deterministically computes keyframe generation mode and denoise level from scene state:
    - Initial scene -> T2I composition at 0.70 denoise
    - Real posture transition (pose_changed=True) -> regenerate at 0.85 denoise
    - Framing/camera size change with same pose (e.g. medium -> close-up) -> crop_zoom at 0.30 denoise
    - Same pose, same framing -> subtle motion continuity at 0.35 denoise
    """
    if prev is None:
        return {"mode": "t2i", "denoise": 0.70}

    # Explicit pose transition flag from compiler
    if cur.get("pose_changed", True):
        return {"mode": "regenerate", "denoise": 0.85}

    cur_size = cur.get("camera_size")
    prev_size = prev.get("camera_size") if prev else None

    # Framing cut while posture remains identical
    if cur_size and prev_size and cur_size != prev_size:
        return {"mode": "crop_zoom", "denoise": 0.30}

    # Same pose and same framing: minimal denoise for stability
    return {"mode": "regenerate", "denoise": 0.35}


class SceneManager:
    plan_keyframe = staticmethod(plan_keyframe)

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
    def build_motion_prompt(scene, scene_state=None, states=None):
        description = scene.get("description", "")
        action = scene.get("action", "")
        camera = scene.get("camera", "")
        lighting = scene.get("lighting", "")

        act = scene.get("act")
        if act:
            is_sexual = act in {"oral", "vaginal", "anal"}
        else:
            # Fallback when 'act' field is absent (for legacy tests/plans)
            is_sexual = any(kw in (description + action).lower() for kw in [
                "sex", "fuck", "penetrat", "oral", "blowjob", "nude", "naked",
                "breast", "pussy", "cock", "penis", "dick", "vagina", "orgasm", "cum", "thrust"
            ])

        motion_intensity = scene.get("motion_intensity", "")
        primary_motion = scene.get("primary_motion", "")
        hand_positions = scene.get("hand_positions", "")

        # Extract structured motion object if present, or generate prompt string
        if scene.get("motion"):
            motion_prompt_text = motion_to_prompt(scene["motion"])
        else:
            motion_prompt_text = scene.get("primary_motion", scene.get("video_prompt", ""))

        clothing_continuity_section = ""
        if states:
            clothing_continuity_section = f"""
CLOTHING CONTINUITY PER PERSON:
{per_character_clothing_block(states)}
"""

        base = f"""
Animate this input keyframe with smooth, continuous, realistic physical motion.

Motion Intensity & Rhythm: {motion_intensity}
Structured Kinetic Motion: {motion_prompt_text}
Primary Motion Focus: {primary_motion}
Hand Placement Anchor: {hand_positions}
{clothing_continuity_section}
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

        tracker = ContinuityTracker(story_plan.get("characters", []))

        # Fast-forward tracker silently if starting from a resumed scene
        if start_scene > 1:
            for s in scenes:
                if int(s.get("order", 0)) < start_scene:
                    tracker.advance(s)
            prev_order = start_scene - 1
            if prev_order >= 1 and prev_order <= len(scenes):
                prev_scene_obj = scenes[prev_order - 1]
                prev_sid = prev_scene_obj.get("id", f"scene_{prev_order:03d}")
                candidate_frame = job_dir / "scenes" / prev_sid / "last_frame.png"
                if candidate_frame.exists():
                    previous_last_frame = candidate_frame

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
            # Track character wardrobe and nudity state via ContinuityTracker
            # --------------------------------------------------
            states = tracker.advance(scene)
            is_nude_scene = any(cs.nude or not cs.bottom_on for cs in states.values())

            # --------------------------------------------------
            # Collect genital references ONLY if scene has exposed lower body
            # --------------------------------------------------
            genital_refs = []
            if is_nude_scene:
                for char_id in scene_character_ids:
                    cs = states[char_id]
                    if cs.bottom_on:           # lower body covered, no anatomy reference needed
                        continue
                    genital_refs.extend(self.get_or_generate_genital_refs(
                        character_appearance=character_map[char_id].get("appearance", ""),
                        gender=cs.gender,
                        character_id=char_id,
                        output_dir=job_dir / "genital_refs",
                    ))
                genital_refs = genital_refs[:2]
            else:
                logger.info(f"[GENITAL REF] Scene {scene_id} is not sexual/nude -> skipping genital references.")

            # Determine adaptive denoise level & setup stage 1
            prev_scene_dict = scenes[order - 2] if order > 1 and len(scenes) >= (order - 1) else None
            kf_plan = plan_keyframe(scene, prev_scene_dict)
            active_denoise = kf_plan["denoise"]
            plan_mode = kf_plan["mode"]

            # Maintain has_pose_transition flag for logging and metadata
            has_pose_transition = scene.get("pose_changed")
            if has_pose_transition is None:
                has_pose_transition = self.detect_pose_or_angle_transition(
                    scene=scene,
                    previous_scene_state=previous_scene_state,
                    previous_character_states=previous_character_states,
                )
                if has_pose_transition:
                    active_denoise = 0.85

            if previous_last_frame is None or plan_mode == "t2i":
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
                print()
                print("=" * 80)
                print(
                    f"SCENE {order}: USING PREVIOUS VIDEO LAST FRAME "
                    f"(mode={plan_mode}, denoise={active_denoise}, pose_transition={has_pose_transition})"
                )
                print("=" * 80)
                print("Previous last frame:", previous_last_frame)

                if plan_mode == "crop_zoom":
                    try:
                        from PIL import Image
                        with Image.open(previous_last_frame) as img:
                            w, h = img.size
                            crop_w, crop_h = int(w * 0.75), int(h * 0.75)
                            left = (w - crop_w) // 2
                            top = (h - crop_h) // 2
                            cropped = img.crop((left, top, left + crop_w, top + crop_h))
                            resized = cropped.resize((w, h), Image.Resampling.LANCZOS)
                            resized.save(composition_path)
                    except Exception as e:
                        logger.warning(f"PIL crop_zoom fallback to direct copy: {e}")
                        shutil.copy2(previous_last_frame, composition_path)
                else:
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
                c_gender = states[char_id].gender if char_id in states else "female"
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
                if c_gender == "female":
                    image_mapping_lines.append(
                        f"  -> STRICT GENDER ANATOMY LOCK FOR '{c_name.upper()}' (FEMALE): "
                        f"Render feminine chest, female waist, and female pussy/vulva between thighs. "
                        f"ABSOLUTELY NEVER attach a male penis, cock, or male testicles to '{c_name}' under any circumstances!"
                    )
                elif c_gender == "male":
                    image_mapping_lines.append(
                        f"  -> STRICT GENDER ANATOMY LOCK FOR '{c_name.upper()}' (MALE): "
                        f"Render masculine torso, flat male chest, and male penis/cock attached only at male pelvis. "
                        f"ABSOLUTELY NEVER render female sports bras, tops, thongs, or female vulva on '{c_name}'!"
                    )

            image_mapping_lines.append("""
=========================================================
CRITICAL CLOTHING CONTINUITY LOCK
=========================================================
- Maintain 100% strict clothing state continuity from Image 1 and character reference.
- IF THE SCENE IS NUDE / UNCLOTHED: Keep character(s) 100% fully nude with complete bare skin. DO NOT invent white tops, sports bras, thongs, wrist wraps, or random underwear on nude bodies unless explicitly instructed in scene description.
- IF THE SCENE IS CLOTHED: Keep their exact canonical outfit from the character reference sheet.
""")

            image_mapping_lines.append("""
=========================================================
CRITICAL POSE LOCK (PREVENT POSE DRIFT)
=========================================================
- Maintain exact body posture, leg positioning, and hip elevation from Image 1 base.
- If female character is on all fours, keep her on all fours with knees spread, back arched, and ass raised.
- If male character is kneeling behind her, keep him kneeling directly behind her hips.
- Do NOT straighten her back, do NOT drop her hips, do NOT stand the male character up.
- Keep hands anchored in their exact described position without floating, jumping, or shifting randomly.
""")

            image_mapping_lines.append("""
=========================================================
POSE & REAR-ENTRY PENETRATION ALIGNMENT LOCK
=========================================================
- For Doggy / Rear-Entry scenes: Female character is on all fours or bending forward with hips raised. Male character kneels or stands directly behind her. Both legs of female character must remain naturally positioned on mattress without twisting, flipping, or distorted hips.
- ACCURATE PENETRATION ALIGNMENT: Male penis connects directly into female pussy entrance at the correct anatomical location from behind or in front. Penis must NOT be positioned above hips, floating in air, on thighs, or sticking out of wrong anatomical places.
- PANEL SELECTION: Use RIGHT PANEL (Rear/Back View) of Image 2 for rear/doggy shots, CENTER PANEL (3/4 Side View) for side shots, and LEFT PANEL (Front View) for front shots.
""")

            image_mapping_lines.append("""
=========================================================
CRITICAL CONTACT & ANATOMY LOCK (NO BLACK HOLES / NO VOIDS)
=========================================================
- Solid, unbroken, continuous skin-to-skin contact between bodies.
- ABSOLUTELY NO black holes, dark circular voids, missing flesh, hollow skin, or dark cutout spots in contact zones.
- Skin must be continuous, smooth, realistic, and fully opaque across all thighs, hips, waist, and genitalia contact areas.
- No shadow distortion turning skin into empty dark holes or black background gaps.
""")

            image_mapping_block = "\n".join(image_mapping_lines)
            clothing_block = per_character_clothing_block(states)
            anatomy_rules = anatomy_lock(states)

            keyframe_prompt = f"""
{image_mapping_block}

=========================================================
SCENE PROMPT & CONTINUITY CONTEXT
=========================================================
CLOTHING PER PERSON:
{clothing_block}

ANATOMY RULES:
{anatomy_rules}

Pose Type: {scene.get('pose_type', '')}
Pose Lock: {scene.get('pose_lock', '')}
Hand Positions: {scene.get('hand_positions', '')}
Genital Alignment: {scene.get('genital_alignment', '')}
Contact Zone: {scene.get('contact_zone', '')}
Facial Expression: {scene.get('expression', '')}
Penetration State: {scene.get('penetration_state', '')}
Primary Motion: {scene.get('primary_motion', '')}
Motion Intensity: {scene.get('motion_intensity', '')}
Previous Scene Context: {scene.get('previous_scene_context', 'None (Scene 1)')}

Scene description:
{scene.get('description', '')}

Action:
{scene.get('action', '')}

Camera:
{scene.get('camera', '')}

Lighting:
{scene.get('lighting', '')}

Character continuity state:
{json.dumps(scene_state.get('character_current_states', {}), ensure_ascii=False, indent=2)}

Maintain character identity, body pose, and clothing from the input keyframe.
Refine genitalia and contact areas in Stage B if applicable.

=========================================================
CRITICAL CONTACT & ANATOMY LOCK (NO BLACK HOLES / NO VOIDS)
=========================================================
- Solid, unbroken, continuous skin-to-skin contact between bodies.
- ABSOLUTELY NO black holes, dark circular voids, missing flesh, hollow skin, or dark cutout spots in contact zones.
- Skin must be continuous, smooth, realistic, and fully opaque across all thighs, hips, waist, and genitalia contact areas.
- No shadow distortion turning skin into empty dark holes or black background gaps.
"""
            # is_nude_scene is determined deterministically above by ContinuityTracker states

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

            # Append context blocks to keyframe_prompt
            if human_count_block:
                keyframe_prompt += f"\n\n{human_count_block}"
            if nudity_instruction:
                keyframe_prompt += f"\n\n{nudity_instruction}"
            if pose_transition_instruction:
                keyframe_prompt += f"\n\n{pose_transition_instruction}"

            negative = dynamic_negative(states)

            # --------------------------------------------------
            # Stage 2: Keyframe Generation - Two-Pass Workflow
            # --------------------------------------------------
            # Stage A: Identity, Body Pose & Clothing (no genital references)
            # --------------------------------------------------
            stage_a_keyframe_path = scene_dir / "stage_a_keyframe.png"

            keyframe_a_result = self.keyframe_generator.generate(
                scene_image=base_image,
                character_references=character_refs,
                prompt=keyframe_prompt,
                output_path=stage_a_keyframe_path,
                seed=2000003 + order * 7919,
                denoise=active_denoise,
                genital_references=[],
                negative_prompt=negative,
                anatomy_rules=anatomy_rules,
            )

            # --------------------------------------------------
            # Stage B: Anatomy Refinement (conditional via needs_anatomy_pass)
            # --------------------------------------------------
            if needs_anatomy_pass(scene.get("act", "none"), states) and genital_refs and hasattr(self.keyframe_generator, "generate_stage_b"):
                # Call Stage B to refine genitalia and contact areas
                keyframe_b_result = self.keyframe_generator.generate_stage_b(
                    stage_a_image=stage_a_keyframe_path,
                    genital_references=genital_refs,
                    prompt=keyframe_prompt,
                    output_path=keyframe_path,
                    seed=2000003 + order * 7919 + 1,
                    denoise=0.25,
                    negative_prompt=negative,
                    anatomy_rules=anatomy_rules,
                )
                keyframe_result = keyframe_b_result
            else:
                # Copy Stage A keyframe directly (no anatomy refinement needed)
                shutil.copy2(stage_a_keyframe_path, keyframe_path)
                keyframe_result = {
                    "prompt_id": keyframe_a_result.get("prompt_id"),
                    "client_id": keyframe_a_result.get("client_id"),
                    "output": str(keyframe_path),
                    "base_image": str(stage_a_keyframe_path),
                    "character_references": keyframe_a_result.get("character_references", []),
                    "genital_references": keyframe_a_result.get("genital_references", []),
                    "denoise": active_denoise,
                }

            # Stage 3: Wan 2.2 Video Generation
            print()
            print("=" * 80)
            print(f"SCENE {order}: GENERATING WAN 2.2 VIDEO")
            print("=" * 80)

            motion_prompt = self.build_motion_prompt(
                scene,
                scene_state=scene_state,
                states=states,
            )

            # Auto-detect motion LoRA category (e.g. anal motion)
            motion_obj = scene.get("motion")
            motion_cat = ""
            if isinstance(motion_obj, dict):
                motion_cat = str(motion_obj.get("category", "")).lower()
            elif hasattr(motion_obj, "category"):
                motion_cat = str(getattr(motion_obj, "category", "")).lower()

            if not motion_cat:
                if scene.get("act") == "anal":
                    motion_cat = "anal"

            motion_lora_name = None
            if motion_cat == "anal":
                motion_lora_name = "wan2.2_motion_anal_v1/wan2.2_motion_anal_v1.safetensors"

            video_result = self.video_generator.generate(
                keyframe=keyframe_path,
                motion_prompt=motion_prompt,
                output_path=video_path,
                seed=3000003 + order * 7919,
                motion_lora_name=motion_lora_name,
                motion_lora_strength=0.65,
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

    def _normalize_gender(self, gender: str) -> str:
        """Return 'male', 'female', or 'unknown' from any common variant."""
        if not gender:
            return "unknown"
        g = str(gender).strip().lower()

        import re
        # Check female first to prevent 'male' matching inside 'female' or 'he' inside 'she'
        if re.search(r"\b(female|woman|women|girl|lady|feminine|she|her|hers)\b", g) or g in ["f", "♀"]:
            return "female"
        if re.search(r"\b(male|man|men|guy|boy|masculine|he|him|his|gentleman)\b", g) or g in ["m", "♂"]:
            return "male"

        return "unknown"

    def get_or_generate_genital_refs(self, character_id, character_appearance, gender, output_dir):
        """
        Generate genital close-up reference once per character based on gender and reuse it.
        Returns a list of paths containing [ref_path] or empty list.
        """
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        normalized = self._normalize_gender(gender)
        is_male = normalized == "male"

        # Fallback: if we couldn't determine gender, don't generate anything
        if normalized == "unknown":
            logger.warning(
                f"Could not determine gender for character '{character_id}' "
                f"(got '{gender}'). Skipping genital reference generation."
            )
            return []

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
            logger.info(f"Generating genital reference for character '{character_id}' ({gender} → {normalized})...")
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