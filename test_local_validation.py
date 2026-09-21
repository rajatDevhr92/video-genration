import os
import sys
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_validation")

def main():
    logger.info("Running local verification checks...")
    
    # 1. Test app imports
    try:
        from app.scene_manager import SceneManager
        from app.keyframe_generator import KeyframeGenerator
        from app.story_planner import StoryPlanner
        from app.story_schema import StoryPlanSchema
        from app.config import OUTPUT_DIR, COMFYUI_URL
        from app.video_assembler import assemble_story_video
        from app.audio_generator import generate_story_audio, resolve_character_voice
        from app.audio_planner import AudioPlanner
        logger.info("✅ Core modules imported successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to import core modules: {e}")
        sys.exit(1)

    # 2. Test voice resolution & gender override logic
    elena_char = {"id": "character_001", "name": "Elena", "appearance": "Adult woman, 28 years old"}
    john_char = {"id": "character_002", "name": "John", "appearance": "Adult man, 30 years old"}

    elena_voice = resolve_character_voice("character_001", elena_char)
    john_voice = resolve_character_voice("character_002", john_char)

    # Test mismatch override: if LLM assigned male voice 'onyx' to female character Elena
    elena_mismatch_voice = resolve_character_voice("character_001", elena_char, requested_voice="onyx")

    assert elena_voice in ["nova", "coral", "shimmer"], f"Female character got non-female voice: {elena_voice}"
    assert john_voice in ["onyx", "echo", "ash"], f"Male character got non-male voice: {john_voice}"
    assert elena_mismatch_voice in ["nova", "coral", "shimmer"], f"Gender mismatch failed to override: {elena_mismatch_voice}"
    logger.info(f"✅ Voice resolution & gender override passed! Female='{elena_voice}', Male='{john_voice}', Overridden Female='{elena_mismatch_voice}'")

    # 3. Test pose transition detection logic & SceneManager dependency injection
    sm = SceneManager(
        keyframe_generator=None,
        video_generator=None,
        composition_generator=None,
    )

    prev_state = {"scene_id": "scene_001", "location": "location_001"}
    prev_char_states = {"char_001": {"current_pose": "standing upright"}}

    test_scene_pose_shift = {
        "id": "scene_003",
        "description": "Elena climbs onto the bed nude, settling back against pillows.",
        "action": "Elena lies back, thighs parting slowly.",
        "camera": "Over-the-shoulder shot.",
        "location_id": "location_001"
    }

    pose_transition = sm.detect_pose_or_angle_transition(
        scene=test_scene_pose_shift,
        previous_scene_state=prev_state,
        previous_character_states=prev_char_states
    )
    assert pose_transition is True, "Pose shift scene failed to detect pose transition!"
    logger.info("✅ Pose transition detector test passed.")

    # 4. Test motion prompt generation for smooth skin, anatomical detail, and character body separation
    motion_prompt = sm.build_motion_prompt(test_scene_pose_shift)
    assert "CHARACTER BODY SEPARATION" in motion_prompt, "Motion prompt missing character body separation!"
    assert "smooth skin texture" in motion_prompt.lower(), "Motion prompt missing smooth skin directive!"
    assert "flawless" in motion_prompt.lower(), "Motion prompt missing flawless skin directive!"
    logger.info("✅ Adult visual detail & character body separation prompt tests passed.")

    logger.info("🎉 ALL LOCAL VERIFICATION CHECKS PASSED!")

if __name__ == "__main__":
    main()

