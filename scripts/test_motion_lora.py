import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.video_generator import VideoGenerator


def test_motion_lora(
    keyframe_path: str,
    output_path: str = "output/test_motion_lora_result.mp4",
    lora_name: str = "wan2.2_multi_motion_v1.safetensors",
    lora_strength: float = 0.65,
    prompt: str = None,
    seed: int = 42,
):
    """
    Runs a test video generation using the trained Multi-Category Motion LoRA.
    """
    keyframe = Path(keyframe_path)
    if not keyframe.exists():
        print(f"Error: Keyframe image not found at {keyframe}")
        return

    # Resolve LoRA file path in ComfyUI models directory
    loras_dir = Path("/workspace/ComfyUI/models/loras")
    
    # Check if lora file exists or is inside subfolder
    subfolder_lora = loras_dir / lora_name.replace(".safetensors", "") / lora_name
    root_lora = loras_dir / lora_name

    resolved_lora_name = lora_name
    if subfolder_lora.exists():
        resolved_lora_name = f"{lora_name.replace('.safetensors', '')}/{lora_name}"
    elif root_lora.exists():
        resolved_lora_name = lora_name

    video_gen = VideoGenerator()

    if not prompt:
        prompt = (
            "motion_lora, cat_penetration, pose_doggy, rear_angle_shot, "
            "continuous rhythmic pelvic thrusting, steady cadence, "
            "solid continuous skin contact, 16fps"
        )

    print("=" * 70)
    print("TESTING TRAINED MOTION LORA")
    print("=" * 70)
    print(f"Keyframe:      {keyframe}")
    print(f"LoRA Name:     {resolved_lora_name}")
    print(f"LoRA Strength: {lora_strength}")
    print(f"Motion Prompt: {prompt}")
    print("=" * 70)

    result = video_gen.generate(
        keyframe=keyframe,
        motion_prompt=prompt,
        output_path=output_path,
        seed=seed,
        motion_lora_name=resolved_lora_name,
        motion_lora_strength=lora_strength,
    )

    print(f"\nTest generation complete! Video saved to: {result['output']}")


def main():
    parser = argparse.ArgumentParser(description="Test trained Wan 2.2 Motion LoRA")
    parser.add_argument("keyframe", help="Path to input keyframe image")
    parser.add_argument("--output", default="output/test_motion_result.mp4", help="Output video path")
    parser.add_argument("--lora-name", default="wan2.2_multi_motion_v1.safetensors", help="LoRA model filename")
    parser.add_argument("--strength", type=float, default=0.65, help="LoRA strength (0.0 - 1.0)")
    parser.add_argument("--prompt", default=None, help="Custom motion prompt")
    parser.add_argument("--seed", type=int, default=42, help="Noise seed")

    args = parser.parse_args()
    test_motion_lora(
        keyframe_path=args.keyframe,
        output_path=args.output,
        lora_name=args.lora_name,
        lora_strength=args.strength,
        prompt=args.prompt,
        seed=args.seed,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_motion_lora.py <path_to_keyframe_image> [--prompt <text>] [--strength 0.65]")
    else:
        main()

