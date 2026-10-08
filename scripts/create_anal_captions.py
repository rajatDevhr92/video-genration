import sys
from pathlib import Path


def create_anal_captions(
    dataset_dir: str = "/workspace/dataset/anal_motion",
    trigger_word: str = "motion_anal_thrust_action"
):
    """
    Creates matching .txt caption files for every .mp4 video clip in anal motion dataset.
    """
    path = Path(dataset_dir)
    if not path.exists():
        print(f"Error: Dataset directory does not exist: {path}")
        return

    caption_text = (
        f"{trigger_word}, explicit adult intimate scene, "
        "continuous rhythmic anal penetration motion, focused body movement, "
        "solid continuous skin contact, 16fps"
    )

    mp4_files = sorted(list(path.glob("*.mp4")))
    if not mp4_files:
        print(f"No .mp4 video files found in {path}")
        return

    created_count = 0
    for video_file in mp4_files:
        txt_file = video_file.with_suffix(".txt")
        txt_file.write_text(caption_text, encoding="utf-8")
        print(f"Created caption for [{video_file.name}] -> [{txt_file.name}]")
        created_count += 1

    print(f"\nSuccessfully generated {created_count} caption files in {path}")


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/dataset/anal_motion"
    create_anal_captions(target_dir)

