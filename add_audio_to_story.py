import argparse
import json
from pathlib import Path

try:
    from app.audio_generator import AudioGenerator
except ImportError:
    from app import AudioGenerator


def main():

    parser = argparse.ArgumentParser(
        description="Generate audio and mux it into story videos."
    )

    parser.add_argument(
        "job_dir",
        help="Path to story job directory",
    )

    parser.add_argument(
        "--start-scene",
        type=int,
        default=1,
        help="First scene to process.",
    )

    parser.add_argument(
        "--end-scene",
        type=int,
        default=None,
        help="Last scene to process.",
    )

    args = parser.parse_args()

    job_dir = Path(args.job_dir)

    plan_path = job_dir / "story_plan.json"

    if not plan_path.exists():
        raise RuntimeError(
            f"story_plan.json not found: {plan_path}"
        )

    with open(
        plan_path,
        "r",
        encoding="utf-8",
    ) as f:
        plan = json.load(f)

    scenes = plan.get("scenes", [])

    if not scenes:
        raise RuntimeError(
            "No scenes found in story_plan.json"
        )

    start_scene = args.start_scene
    end_scene = (
        args.end_scene
        if args.end_scene is not None
        else len(scenes)
    )

    selected_scenes = [
        scene
        for scene in scenes
        if start_scene <= scene.get("order", 0) <= end_scene
    ]

    if not selected_scenes:
        raise RuntimeError(
            f"No scenes found between "
            f"{start_scene} and {end_scene}."
        )

    generator = AudioGenerator()

    print()
    print("=" * 80)
    print("AUDIO GENERATION")
    print("=" * 80)
    print("Job:", job_dir)
    print("Scenes:", start_scene, "->", end_scene)

    for scene in selected_scenes:

        scene_id = scene["id"]

        scene_dir = (
            job_dir
            / "scenes"
            / scene_id
        )

        video = scene_dir / "video.mp4"
        audio = scene_dir / "audio.wav"
        video_audio = (
            scene_dir
            / "video_with_audio.mp4"
        )

        if not video.exists():
            raise RuntimeError(
                f"Missing video: {video}"
            )

        print()
        print("=" * 80)
        print(f"AUDIO SCENE: {scene_id}")
        print("=" * 80)

        audio_plan = scene.get(
            "audio",
            {},
        )

        if not isinstance(audio_plan, dict):
            raise RuntimeError(
                f"Scene '{scene_id}' has invalid audio data. "
                f"Expected object, got {type(audio_plan).__name__}."
            )

        character_map = {
            c["id"]: c for c in plan.get("characters", [])
        }

        generator.create_scene_audio(
            audio_plan=audio_plan,
            output_path=audio,
            character_map=character_map,
        )

        generator.mux_video_audio(
            video_path=video,
            audio_path=audio,
            output_path=video_audio,
        )

        print("Audio:", audio)
        print(
            "Video + audio:",
            video_audio,
        )

        metadata_path = (
            scene_dir / "metadata.json"
        )

        if metadata_path.exists():

            with open(
                metadata_path,
                "r",
                encoding="utf-8",
            ) as f:
                metadata = json.load(f)

            metadata["audio"] = str(audio)

            metadata["video_with_audio"] = str(
                video_audio
            )

            with open(
                metadata_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    metadata,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

    print()
    print("=" * 80)
    print("AUDIO GENERATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()