#!/usr/bin/env python3
"""
cloning/seam_report.py - Seam diagnostic tool for multi-chunk cloned anime videos.

Given a completed or in-progress job folder, this script:
1. Loads all sequential chunk_XXXX.mp4 files in order.
2. For each chunk boundary (N -> N+1), extracts the last frame of chunk N
   and the first frame of chunk N+1.
3. Computes:
   - Mean absolute color difference (0 to 255) in RGB space.
   - Structural Similarity (SSIM, 0.0 to 1.0).
   - Evaluates against configurable thresholds (SSIM threshold & color delta threshold).
4. Saves high-resolution side-by-side comparison images into <job_dir>/seams/seam_XXXX_XXXX.png.
5. Prints a detailed boundary-by-boundary report and summary.

Usage:
  python3 -m cloning.seam_report /path/to/job_dir [--color-thresh 35.0] [--ssim-thresh 0.55]
  python3 cloning/seam_report.py /workspace/story-video-engine/output/clone_e5bbe5e24089
"""
import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Any

# Threshold defaults (can be overridden via CLI flags)
DEFAULT_SSIM_THRESH = 0.55
DEFAULT_COLOR_THRESH = 35.0


def compute_ssim_numpy(img1: Any, img2: Any) -> float:
    """Computes Structural Similarity Index (SSIM) between two 2D arrays."""
    try:
        import cv2
        import numpy as np

        C1 = (0.01 * 255.0) ** 2
        C2 = (0.03 * 255.0) ** 2

        f1 = img1.astype(np.float32)
        f2 = img2.astype(np.float32)

        mu1 = cv2.GaussianBlur(f1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(f2, (11, 11), 1.5)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(f1 * f1, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(f2 * f2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(f1 * f2, (11, 11), 1.5) - mu1_mu2

        num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = num / (den + 1e-7)
        return float(np.clip(np.mean(ssim_map), 0.0, 1.0))
    except Exception:
        return 0.0


def extract_first_frame(video_path: Path):
    """Extracts first frame as BGR numpy array using cv2."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Could not read first frame from {video_path}")
    return frame


def extract_last_frame(video_path: Path):
    """Extracts last frame as BGR numpy array using cv2."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    if not ret or frame is None:
        # Fallback: rewind and read until end
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        last = None
        while True:
            r, f = cap.read()
            if not r:
                break
            last = f
        frame = last
    cap.release()
    if frame is None:
        raise RuntimeError(f"Could not read last frame from {video_path}")
    return frame


def find_chunk_files(job_dir: Path) -> List[Path]:
    """Finds all final completed chunk_XXXX.mp4 files in order."""
    search_dirs = [job_dir / "chunks", job_dir]
    chunks = []
    chunk_pattern = re.compile(r"^chunk_(\d{4})\.mp4$")

    for d in search_dirs:
        if d.exists() and d.is_dir():
            for f in d.glob("chunk_*.mp4"):
                m = chunk_pattern.match(f.name)
                # Exclude intermediate candidates: chunk_XXXX_src, chunk_XXXX_control, chunk_XXXX_att
                if m and f.stat().st_size > 1000:
                    chunks.append((int(m.group(1)), f))
            if chunks:
                break

    chunks.sort(key=lambda x: x[0])
    return [c[1] for c in chunks]


def analyze_seams(
    job_dir: Path,
    ssim_thresh: float = DEFAULT_SSIM_THRESH,
    color_thresh: float = DEFAULT_COLOR_THRESH,
) -> int:
    """Analyzes boundaries between sequential chunks and saves side-by-side seam images."""
    import cv2
    import numpy as np

    job_dir = Path(job_dir).resolve()
    if not job_dir.exists():
        print(f"Error: Job directory does not exist: {job_dir}")
        return 1

    chunks = find_chunk_files(job_dir)
    if len(chunks) < 2:
        print(f"Found {len(chunks)} completed chunk(s) in {job_dir}.")
        print("At least 2 chunks are required to evaluate boundary seams.")
        if chunks:
            print(f"  Available: {chunks[0].name}")
        return 0

    seams_dir = job_dir / "seams"
    seams_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"  SEAM REPORT: {job_dir.name}")
    print(f"  Total Chunks: {len(chunks)}  |  Total Boundaries: {len(chunks) - 1}")
    print(f"  Thresholds: SSIM >= {ssim_thresh:.2f} (PASS), Mean Color Delta <= {color_thresh:.1f} (PASS)")
    print("=" * 80)
    print(f"{'Boundary':<18} | {'SSIM':<12} | {'Color Delta':<14} | {'Status':<8} | {'Image Output'}")
    print("-" * 80)

    total_seams = len(chunks) - 1
    passed_count = 0
    warning_count = 0

    for i in range(total_seams):
        chunk_a = chunks[i]
        chunk_b = chunks[i + 1]

        try:
            frame_a = extract_last_frame(chunk_a)
            frame_b = extract_first_frame(chunk_b)
        except Exception as err:
            print(f"chunk_{i:04d} -> chunk_{i+1:04d} | ERROR extracting frames: {err}")
            warning_count += 1
            continue

        # Resize frame_b to match frame_a dimensions if needed
        h_a, w_a = frame_a.shape[:2]
        h_b, w_b = frame_b.shape[:2]
        if (h_a, w_a) != (h_b, w_b):
            frame_b = cv2.resize(frame_b, (w_a, h_a), interpolation=cv2.INTER_AREA)

        # 1. Structural similarity (SSIM) on grayscale
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        ssim_val = compute_ssim_numpy(gray_a, gray_b)

        # 2. Mean color difference across RGB channels (0 - 255)
        diff_rgb = np.abs(frame_a.astype(np.float32) - frame_b.astype(np.float32))
        color_delta = float(np.mean(diff_rgb))

        ssim_pass = ssim_val >= ssim_thresh
        color_pass = color_delta <= color_thresh
        is_pass = ssim_pass and color_pass

        if is_pass:
            status_str = "PASS"
            passed_count += 1
        else:
            status_str = "WARN"
            warning_count += 1

        # 3. Create side-by-side composite visualization
        vis_h = max(240, h_a)
        vis_w = max(426, w_a)
        fa_scaled = cv2.resize(frame_a, (vis_w, vis_h))
        fb_scaled = cv2.resize(frame_b, (vis_w, vis_h))

        # Add top banner for metrics
        banner_h = 44
        composite = np.zeros((vis_h + banner_h, vis_w * 2, 3), dtype=np.uint8)
        composite[banner_h:, :vis_w] = fa_scaled
        composite[banner_h:, vis_w:] = fb_scaled

        # Banner background: dark gray with color-coded status
        composite[:banner_h, :] = (35, 35, 35)
        status_color = (0, 220, 80) if is_pass else (0, 100, 255)

        banner_text = (
            f"Seam {chunk_a.stem} (Last) -> {chunk_b.stem} (First)  |  "
            f"SSIM: {ssim_val:.3f} ({'PASS' if ssim_pass else 'WARN'})  |  "
            f"Color Delta: {color_delta:.1f} ({'PASS' if color_pass else 'WARN'})  |  [{status_str}]"
        )
        cv2.putText(
            composite,
            banner_text,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
            cv2.LINE_AA,
        )

        out_img_name = f"seam_{i:04d}_{i+1:04d}.png"
        out_img_path = seams_dir / out_img_name
        cv2.imwrite(str(out_img_path), composite)

        boundary_label = f"chunk_{i:04d} -> {i+1:04d}"
        ssim_col = f"{ssim_val:.3f} ({'OK' if ssim_pass else 'LOW'})"
        color_col = f"{color_delta:.1f} ({'OK' if color_pass else 'HIGH'})"
        rel_img_path = f"seams/{out_img_name}"

        print(f"{boundary_label:<18} | {ssim_col:<12} | {color_col:<14} | {status_str:<8} | {rel_img_path}")

    print("-" * 80)
    print(f"Summary: {passed_count}/{total_seams} seams PASSED, {warning_count} flagged with WARNING.")
    print(f"Side-by-side diagnostic images saved to: {seams_dir}")
    print("=" * 80)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Analyze visual seams between sequential video chunks.")
    parser.add_argument("job_dir", type=str, help="Path to job directory (e.g. output/clone_e5bbe5e24089)")
    parser.add_argument(
        "--ssim-thresh",
        type=float,
        default=DEFAULT_SSIM_THRESH,
        help=f"Minimum SSIM for seamless boundary (default: {DEFAULT_SSIM_THRESH})",
    )
    parser.add_argument(
        "--color-thresh",
        type=float,
        default=DEFAULT_COLOR_THRESH,
        help=f"Maximum mean color delta for seamless boundary (default: {DEFAULT_COLOR_THRESH})",
    )
    args = parser.parse_args()

    return analyze_seams(
        job_dir=Path(args.job_dir),
        ssim_thresh=args.ssim_thresh,
        color_thresh=args.color_thresh,
    )


if __name__ == "__main__":
    sys.exit(main())
