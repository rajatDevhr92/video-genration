import subprocess
from pathlib import Path


class VideoAssembler:

    @staticmethod
    def assemble(
        job_dir,
        output_path,
    ):
        job_dir = Path(job_dir)
        output_path = Path(output_path)

        scene_videos = sorted(
            job_dir.glob(
                "scenes/scene_*/video_with_audio.mp4"
            ),
            key=lambda p: int(
                p.parent.name.split("_")[-1]
            ),
        )

        if not scene_videos:
            raise RuntimeError(
                "No video_with_audio.mp4 files found."
            )

        concat_file = (
            job_dir / "concat_audio.txt"
        )

        with open(
            concat_file,
            "w",
            encoding="utf-8",
        ) as f:
            for video in scene_videos:
                f.write(
                    f"file '{video.resolve()}'\n"
                )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg assembly failed:\n"
                + result.stderr
            )

        if not output_path.exists():
            raise RuntimeError(
                f"Final output was not created: {output_path}"
            )

        return {
            "output": str(output_path),
            "scene_count": len(scene_videos),
        }