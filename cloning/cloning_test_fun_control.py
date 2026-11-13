"""
cloning_test_fun_control.py - Standalone test script for Wan 2.2 14B Fun Control video pipeline.

Runs the complete 2D anime conversion on ONE 5-second chunk (16 FPS, 81 frames):
1. Keyframe extraction & character identity generation.
2. Pose + Depth control video generation.
3. Wan 2.2 Fun Control generation (from step 0, no LoRA, 20 steps).
4. Automatic quality gate evaluation (Farneback optical flow motion match, CLIP anime check, flicker).
5. Prints the complete quality report.

Usage:
  python3 -m cloning.cloning_test_fun_control [path_to_video.mp4]
"""
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    CHUNK_DURATION_SEC,
    CHUNK_TARGET_FRAMES,
    CONTROL_VIDEO_MODE,
    FUN_CONTROL_NEGATIVE_PROMPT,
    FUN_CONTROL_POSITIVE_PROMPT,
    FUN_CONTROL_STEPS,
    OUTPUT_DIR,
    QWEN_KEYFRAME_NEGATIVE,
    QWEN_KEYFRAME_PROMPT,
)
from cloning.wan_video_cloner import WanVideoCloner


def main():
    print("=" * 70)
    print("  WAN 2.2 14B FUN CONTROL 1-CHUNK PIPELINE TEST")
    print("=" * 70)

    # 1. Resolve source video
    source_video = None
    if len(sys.argv) > 1:
        custom_path = Path(sys.argv[1])
        if custom_path.exists():
            source_video = custom_path

    if source_video is None:
        candidates = [
            PROJECT_ROOT / "output" / "chunk_0001_src.mp4",
            PROJECT_ROOT / "input" / "cloning" / "input_bj_bed_cu_01.mp4",
            PROJECT_ROOT / "output" / "clone_75b227a23126" / "input_bj_bed_cu_01.mp4",
        ]
        for c in candidates:
            if c.exists() and c.stat().st_size > 10000:
                source_video = c
                break

    if source_video is None or not source_video.exists():
        print("ERROR: No test source video found. Please provide an MP4 file:")
        print("  python3 -m cloning.cloning_test_fun_control /path/to/test_clip.mp4")
        sys.exit(1)

    print(f"Test Video Source: {source_video} ({source_video.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Setup isolated test output directory
    test_job_dir = OUTPUT_DIR / "test_fun_control_run"
    test_job_dir.mkdir(parents=True, exist_ok=True)
    test_output_video = test_job_dir / "test_output_anime.mp4"

    # 3. Instantiate cloner
    cloner = WanVideoCloner()

    prompt_override = None  # Dict of optional overrides if passed
    resolved_qwen = (prompt_override.get("qwen_keyframe") if prompt_override else None) or QWEN_KEYFRAME_PROMPT
    qwen_source = "OVERRIDE" if (prompt_override and "qwen_keyframe" in prompt_override) else "CONFIG (QWEN_KEYFRAME_PROMPT)"

    resolved_qwen_neg = (prompt_override.get("qwen_negative") if prompt_override else None) or QWEN_KEYFRAME_NEGATIVE
    qwen_neg_source = "OVERRIDE" if (prompt_override and "qwen_negative" in prompt_override) else "CONFIG (QWEN_KEYFRAME_NEGATIVE)"

    resolved_fc_pos = (prompt_override.get("fun_control_positive") if prompt_override else None) or FUN_CONTROL_POSITIVE_PROMPT
    fc_pos_source = "OVERRIDE" if (prompt_override and "fun_control_positive" in prompt_override) else "CONFIG (FUN_CONTROL_POSITIVE_PROMPT)"

    resolved_fc_neg = (prompt_override.get("fun_control_negative") if prompt_override else None) or FUN_CONTROL_NEGATIVE_PROMPT
    fc_neg_source = "OVERRIDE" if (prompt_override and "fun_control_negative" in prompt_override) else "CONFIG (FUN_CONTROL_NEGATIVE_PROMPT)"

    print("-" * 70)
    print("  FINAL PROMPTS CONFIGURED FOR TEST RUN:")
    print(f"  * Qwen Keyframe [{qwen_source}]:\n    {resolved_qwen[:100]}...")
    print(f"  * Qwen Negative [{qwen_neg_source}]:\n    {resolved_qwen_neg}")
    print(f"  * Fun Control Positive [{fc_pos_source}]:\n    {resolved_fc_pos}")
    print(f"  * Fun Control Negative [{fc_neg_source}]:\n    {resolved_fc_neg[:100]}...")
    print("-" * 70)

    print(f"Starting {CHUNK_DURATION_SEC}s ({CHUNK_TARGET_FRAMES} frames) test execution...")
    print(f"Backend: fun_control | Control Mode: {CONTROL_VIDEO_MODE} | Steps: {FUN_CONTROL_STEPS}")

    output_path = cloner.clone_video(
        source_video=source_video,
        character_prompts=[
            "masterpiece 2D Japanese anime woman, Kyoto Animation style, blue hair, expressive anime eyes",
            "masterpiece 2D Japanese anime man, Makoto Shinkai style, neat black hair, handsome anime face",
        ],
        output_video=test_output_video,
        job_dir=test_job_dir,
        seed=42,
        style="anime",
        max_duration=CHUNK_DURATION_SEC,  # Process exactly 1 chunk (10.0s / 161 frames)
        cloning_backend="fun_control",
        control_mode=CONTROL_VIDEO_MODE,
    )

    print("\n" + "=" * 70)
    print("  TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Cloned Output Video: {output_path} ({output_path.stat().st_size} bytes)")

    # 4. Read and display quality gate report
    quality_file = test_job_dir / "chunks" / "chunk_0000_quality.json"
    if quality_file.exists():
        with open(quality_file, "r", encoding="utf-8") as qf:
            q_data = json.load(qf)
        print("\n--- QUALITY GATE REPORT (Chunk 0) ---")
        print(f"Passed Quality Gate: {q_data.get('passed')}")
        print(f"Best Attempt Score: {q_data.get('best_score')}")
        print(f"Number of Attempts: {q_data.get('num_attempts')}")
        for att in q_data.get("attempts", []):
            sc = att.get("scores", {})
            print(f"  Attempt {att.get('attempt')}: overall={sc.get('overall_score')}, motion={sc.get('motion_score')} (passed={sc.get('motion_passed')}), anime={sc.get('anime_score')} (passed={sc.get('anime_passed')}), flicker={sc.get('flicker_score')} (passed={sc.get('flicker_passed')})")
    print("=" * 70)


if __name__ == "__main__":
    main()
