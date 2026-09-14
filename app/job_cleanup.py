import shutil
from pathlib import Path


def cleanup_completed_job(
    job_dir,
    final_video_name="final_with_audio.mp4",
):
    job_dir = Path(job_dir)

    final_video = (
        job_dir
        / "final"
        / final_video_name
    )

    if not final_video.exists():
        raise RuntimeError(
            f"Final video does not exist: {final_video}"
        )

    scenes_dir = job_dir / "scenes"
    references_dir = job_dir / "references"

    if scenes_dir.exists():
        shutil.rmtree(scenes_dir)

    if references_dir.exists():
        shutil.rmtree(references_dir)

    concat_files = [
        job_dir / "concat.txt",
        job_dir / "concat_audio.txt",
    ]

    for path in concat_files:
        if path.exists():
            path.unlink()

    return str(final_video)