import subprocess
from pathlib import Path


def _run(command):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return result


def extract_last_frame(
    video_path,
    output_path,
):
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # Get exact duration
    # ----------------------------------------------------------

    probe = _run([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ])

    if probe.returncode != 0:
        raise RuntimeError(
            "ffprobe failed while reading video duration:\n"
            + probe.stderr
        )

    try:
        duration = float(
            probe.stdout.strip()
        )
    except ValueError:
        raise RuntimeError(
            f"Could not determine video duration. "
            f"ffprobe output: {probe.stdout!r}"
        )

    # Seek slightly before the end rather than exactly at EOF.
    seek_time = max(
        0.0,
        duration - 0.25,
    )

    # ----------------------------------------------------------
    # First extraction attempt
    # ----------------------------------------------------------

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(seek_time),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2",
        "-c:v",
        "png",
        str(output_path),
    ]

    result = _run(command)

    if (
        result.returncode == 0
        and output_path.exists()
        and output_path.stat().st_size > 0
    ):
        return output_path

    # ----------------------------------------------------------
    # Fallback: seek one second before EOF
    # ----------------------------------------------------------

    output_path.unlink(
        missing_ok=True
    )

    fallback = [
        "ffmpeg",
        "-y",
        "-sseof",
        "-1",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2",
        "-c:v",
        "png",
        str(output_path),
    ]

    result = _run(fallback)

    if (
        result.returncode == 0
        and output_path.exists()
        and output_path.stat().st_size > 0
    ):
        return output_path

    raise RuntimeError(
        "FFmpeg could not extract the last frame.\n\n"
        f"Duration: {duration}\n"
        f"First attempt:\n{result.stderr}"
    )