import sys
from pathlib import Path

# Add project root to sys.path to support imports from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.video_generator import VideoGenerator


def test_anal_lora(
    keyframe_path: str,
    output_path: str = "output/test_anal_lora_result.mp4",
    lora_name: str = "wan2.2_motion_anal_v1.safetensors",
    lora_strength: float = 0.65,
    seed: int = 42,
):
    """
    Runs a single video generation test using the newly trained Anal Motion LoRA.
    """
    keyframe = Path(keyframe_path)
    if not keyframe.exists():
        print(f"Error: Keyframe image not found at {keyframe}")
        return

    # Resolve LoRA file path
    loras_dir = Path("/workspace/ComfyUI/models/loras")
    subfolder_lora = loras_dir / "wan2.2_motion_anal_v1" / "wan2.2_motion_anal_v1.safetensors"
    root_lora = loras_dir / "wan2.2_motion_anal_v1.safetensors"

    if subfolder_lora.exists():
        lora_name = "wan2.2_motion_anal_v1/wan2.2_motion_anal_v1.safetensors"
    elif root_lora.exists():
        lora_name = "wan2.2_motion_anal_v1.safetensors"

    video_gen = VideoGenerator()

    prompt = (
        "motion_anal_thrust_action, explicit adult intimate scene, "
        "continuous rhythmic anal penetration motion, kneeling position, "
        "solid continuous skin contact, no black holes, 16fps"
    )

    print(f"Starting test video generation with LoRA [{lora_name}] at strength {lora_strength}...")

    result = video_gen.generate(
        keyframe=keyframe,
        motion_prompt=prompt,
        output_path=output_path,
        seed=seed,
        motion_lora_name=lora_name,
        motion_lora_strength=lora_strength,
    )

    print(f"Test generation complete! Video saved to: {result['output']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_anal_lora.py <path_to_keyframe_image>")
    else:
        test_anal_lora(sys.argv[1])

