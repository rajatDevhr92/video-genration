import sys
from pathlib import Path


def create_caption_for_clip(file_path: Path, trigger_word: str = "motion_lora") -> str:
    name = file_path.stem.lower()
    parent_folder = file_path.parent.name.lower()

    # -------------------------------------------------------------
    # 1. KISSING CATEGORY
    # -------------------------------------------------------------
    if "kiss" in parent_folder or "kiss" in name:
        if "neck" in name:
            act = "neck_kissing, trailing_kisses_along_jawline_and_neck, head_tilted_back"
        elif "bed" in name:
            act = "intimate_kissing, pose_lying_bed, couple_reclining_on_mattress"
        elif "wall" in name:
            act = "intimate_kissing, pose_standing_against_wall, passionate_embrace"
        else:
            act = "intimate_kissing, pose_standing_embrace, arms_wrapped_around_neck"

        cam = "close_up_shot, tight_framing" if "cu" in name else "side_profile_angle, 3/4_view"
        return f"{trigger_word}, cat_foreplay, {act}, {cam}, smooth_sensual_motion, 16fps"

    # -------------------------------------------------------------
    # 2. FOREPLAY / MASSAGE / ORAL TEASING
    # -------------------------------------------------------------
    if "forplay" in parent_folder or "foreplay" in parent_folder:
        if "straddle_face" in name:
            act = "oral_caress, pose_straddling_face, woman_kneeling_over_man_face, hands_on_chest"
        elif "oral" in name or "licking" in name:
            loc = "pose_table_edge" if "table" in name else "pose_reclining_bed"
            act = f"oral_caress, {loc}, woman_lying_back_thighs_parted, gentle_worship"
        elif "breast" in name or "sucking" in name:
            act = "breast_caress, gentle_kissing_and_oral_touch, hands_cupping_skin"
        elif "massage" in name or "pressing" in name or "caress" in name:
            act = "sensual_massage, hands_massaging_bare_chest, fingers_pressing_softly"
        else:
            act = "sensual_foreplay, tender_intimate_touch, warm_lighting"

        cam = "close_up_shot, tight_framing" if "cu" in name else "side_profile_angle, 3/4_view"
        return f"{trigger_word}, cat_foreplay, {act}, {cam}, slow_tender_motion, 16fps"

    # -------------------------------------------------------------
    # 3. LICKING (ORAL CUNNILINGUS)
    # -------------------------------------------------------------
    if "liking" in parent_folder or "licking" in parent_folder or "lick" in name:
        if "side" in name:
            cam = "side_profile_angle, man_head_profile"
        elif "top" in name:
            cam = "high_angle_view, overhead_intimate_perspective"
        elif "hands" in name:
            cam = "front_angle, hands_gently_parting_thighs"
        else:
            cam = "front_angle, smooth_rhythmic_licking_motion"
        return f"{trigger_word}, cat_oral, cunnilingus_oral_worship, close_up_shot, macro_framing, {cam}, continuous_contact, 16fps"

    # -------------------------------------------------------------
    # 4. BLOWJOB (ORAL ACTION)
    # -------------------------------------------------------------
    if "blowjob" in parent_folder or "bj" in name:
        pose = "pose_bed, reclining_on_bed_edge" if "bed" in name else "pose_kneeling, character_kneeling"
        if "top" in name:
            cam = "high_angle_shot, overhead_view"
        elif "cu" in name:
            cam = "close_up_shot, tight_framing"
        elif "side" in name:
            cam = "side_profile_angle"
        else:
            cam = "medium_shot, eye_level"
        return f"{trigger_word}, cat_oral, oral_interaction, {pose}, {cam}, smooth_rhythmic_movement, 16fps"

    # -------------------------------------------------------------
    # 5. COWGIRL / REVERSE COWGIRL
    # -------------------------------------------------------------
    if "cowgirl" in parent_folder or "cowgirl" in name:
        if "reverse" in name:
            pose = "pose_reverse_cowgirl, woman_straddling_facing_away, arched_back, hands_on_knees"
            cam = "rear_angle_shot"
        else:
            pose = "pose_cowgirl, woman_straddling_on_top, upright_torso, hands_on_chest"
            cam = "close_up_shot, tight_framing" if "cu" in name else "front_medium_shot, eye_level"
        return f"{trigger_word}, cat_penetration, {pose}, {cam}, fluid_vertical_rhythmic_motion, solid_contact, 16fps"

    # -------------------------------------------------------------
    # 6. DOGGY / STANDING DOGGY
    # -------------------------------------------------------------
    if "doggy" in parent_folder or "doggy" in name:
        if "standing" in name or "standing" in parent_folder:
            pose = "pose_standing_doggy, woman_standing_bent_over, man_standing_behind"
        elif "prone" in name:
            pose = "pose_doggy_prone, woman_lying_flat_over_pillows, torso_low_hips_raised"
        else:
            pose = "pose_doggy, woman_on_all_fours_arched_back, man_kneeling_behind, hands_on_hips"

        if "cu" in name:
            cam = "close_up_shot, tight_framing"
        elif "side" in name:
            cam = "side_profile_angle, 3/4_view"
        elif "rear" in name:
            cam = "rear_angle_shot"
        else:
            cam = "medium_shot, 3/4_rear_angle"
        return f"{trigger_word}, cat_penetration, {pose}, {cam}, continuous_rhythmic_thrusting, solid_skin_contact, 16fps"

    # -------------------------------------------------------------
    # 7. MISSIONARY
    # -------------------------------------------------------------
    if "missionary" in parent_folder or "missionary" in name:
        setting = "setting_sofa" if "sofa" in name else "setting_bed"
        leg = "pose_missionary_elevated, one_leg_over_shoulder" if ("leg" in name or "shoulder" in name) else "pose_missionary, classic_reclined_posture"
        cam = "close_up_shot, tight_framing" if ("cu" in name or "detail" in name) else "side_profile_angle, 3/4_view"
        return f"{trigger_word}, cat_penetration, {setting}, {leg}, {cam}, chest_to_chest_proximity, steady_rhythmic_cadence, 16fps"

    # -------------------------------------------------------------
    # 8. CLIMAX / FINISH
    # -------------------------------------------------------------
    if "cum" in parent_folder or "climax" in name:
        if "stomach" in name or "belly" in name:
            target = "target_abdomen, bare_torso_skin, resting_on_bed"
        elif "intimate" in name or "vagina" in name:
            target = "target_intimate_area, tight_detail_framing"
        else:
            target = "target_mouth, lips_and_mouth, heavy_breathing"

        cam = "medium_shot" if "medium" in name else "close_up_shot, macro_framing"
        return f"{trigger_word}, cat_climax, {target}, {cam}, fluid_release, sensual_afterglow, 16fps"

    # -------------------------------------------------------------
    # 9. CAMERA TRANSITIONS (ZOOM IN / OUT)
    # -------------------------------------------------------------
    if "zoom" in parent_folder or "camera" in name or "transition" in name:
        if "zoom_out" in name or "pull_back" in name:
            cam_motion = "camera_zoom_out, slow_pull_back, moving_from_close_up_to_wide_shot"
        else:
            cam_motion = "camera_zoom_in, smooth_push_in, moving_from_medium_shot_to_close_up"
        return f"{trigger_word}, dynamic_camera, {cam_motion}, continuous_intimate_action, seamless_transition, 16fps"

    # -------------------------------------------------------------
    # 10. ANAL MOTION
    # -------------------------------------------------------------
    if "anal" in parent_folder or "anal" in name:
        if "expression" in name:
            pose = "close_up_shot, facial_pleasure_reaction, woman_head_and_shoulders, heavy_sensual_breathing"
        elif "cowgirl" in name:
            pose = "pose_cowgirl, woman_straddling_on_top, fluid_vertical_rhythmic_motion"
        elif "macro" in name or "cu" in name:
            pose = "close_up_shot, macro_framing, intimate_contact_zone, deep_deliberate_strokes"
        elif "side" in name:
            pose = "pose_doggy, side_profile_angle, 3/4_view, deep_continuous_rhythmic_thrusts"
        else:
            pose = "pose_doggy, rear_angle_shot, woman_on_all_fours_arched_back, hands_on_hips"
        return f"{trigger_word}, cat_anal_motion, {pose}, solid_skin_contact, 16fps"

    # Fallback generic default
    return f"{trigger_word}, cat_intimate, focused_continuous_motion, solid_skin_contact, 16fps"


def run_batch_captioning(dataset_root: str):
    root = Path(dataset_root)
    if not root.exists():
        print(f"Error: Dataset directory does not exist: {root}")
        return

    mp4_files = sorted(list(root.rglob("*.mp4")))
    if not mp4_files:
        print(f"No .mp4 files found in {root}")
        return

    print(f"Found {len(mp4_files)} clips in {root}")

    created = 0
    for idx, video in enumerate(mp4_files, 1):
        caption_content = create_caption_for_clip(video)
        txt_path = video.with_suffix(".txt")
        txt_path.write_text(caption_content, encoding="utf-8")
        print(f"[{idx}/{len(mp4_files)}] {video.parent.name}/{video.name} -> {caption_content[:60]}...")
        created += 1

    print(f"\nAll {created} caption files generated successfully!")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/workspace/dataset"
    run_batch_captioning(target)

