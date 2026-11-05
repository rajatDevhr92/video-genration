"""
cloning_generator.py - End-to-end face replacement engine for videos.

Pipeline flow:
1. Detects or generates character face reference image(s) using CompositionGenerator (Krea 2)
   if character_prompt is omitted or custom.
2. Extracts audio from source video via FFmpeg.
3. Samples frames from source video, detects faces using InsightFace (SCRFD + ArcFace + genderage).
4. Matches detected faces to target character specifications (by gender, age, embedding similarity).
5. Performs face swap using InSwapper (or InsightFace Swapper) frame-by-frame.
6. Re-stitches frames with original audio using FFmpeg into final MP4.
"""
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

INSWAPPER_MODEL_URL = os.getenv(
    "INSWAPPER_MODEL_URL",
    "https://huggingface.co/ModelsLab/inswapper/resolve/main/inswapper_128.onnx"
)
MODELS_DIR = Path(os.getenv("MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))


def get_inswapper_path() -> Path:
    """Ensure the inswapper_128.onnx model is downloaded and return its path."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "inswapper_128.onnx"
    if not model_path.exists():
        logger.info(f"Downloading InSwapper model to {model_path}...")
        try:
            urllib.request.urlretrieve(INSWAPPER_MODEL_URL, str(model_path))
            logger.info("InSwapper model downloaded successfully.")
        except Exception as e:
            logger.warning(f"Could not auto-download InSwapper model: {e}")
    return model_path


class FaceSwapGenerator:
    def __init__(self, swapper_model_path: Optional[Path] = None):
        self.swapper_model_path = swapper_model_path or get_inswapper_path()
        self._face_app = None
        self._swapper = None

    @property
    def face_app(self):
        if self._face_app is None:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._face_app = app
        return self._face_app

    @property
    def swapper(self):
        if self._swapper is None:
            if not self.swapper_model_path.exists():
                raise FileNotFoundError(
                    f"InSwapper model not found at {self.swapper_model_path}. "
                    "Please download inswapper_128.onnx to models/ folder."
                )
            import insightface
            self._swapper = insightface.model_zoo.get_model(
                str(self.swapper_model_path),
                download=False,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        return self._swapper

    def analyze_source_video(self, video_path: Path, max_sample_frames: int = 15) -> List[Dict[str, Any]]:
        """
        Samples frames across the source video to detect all present characters,
        their genders, average apparent ages, and face embeddings.
        Returns clustered character profiles: [{"cluster_id": 0, "gender": "female", "age": 25, "embedding": ...}]
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        step = max(1, total_frames // max_sample_frames)
        all_detections = []

        for frame_idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            faces = self.face_app.get(frame)
            for face in faces:
                # face.gender: 0 for female, 1 for male in InsightFace
                gender_str = "female" if getattr(face, "gender", 0) == 0 else "male"
                age_val = float(getattr(face, "age", 25))
                embedding = face.normed_embedding if hasattr(face, "normed_embedding") else face.embedding / np.linalg.norm(face.embedding)
                all_detections.append({
                    "frame": frame_idx,
                    "gender": gender_str,
                    "age": age_val,
                    "embedding": embedding,
                    "bbox": face.bbox
                })

        cap.release()
        if not all_detections:
            return []

        # Cluster detections into distinct persons using cosine similarity
        clusters: List[Dict[str, Any]] = []
        for det in all_detections:
            matched = False
            for cl in clusters:
                sim = float(np.dot(cl["centroid_embedding"], det["embedding"]))
                if sim >= 0.45:
                    cl["detections"].append(det)
                    # Update running centroid
                    emb_matrix = np.array([d["embedding"] for d in cl["detections"]])
                    centroid = np.mean(emb_matrix, axis=0)
                    cl["centroid_embedding"] = centroid / np.linalg.norm(centroid)
                    matched = True
                    break
            if not matched:
                clusters.append({
                    "cluster_id": len(clusters),
                    "detections": [det],
                    "centroid_embedding": det["embedding"],
                })

        # Calculate final stats per cluster
        results = []
        for cl in clusters:
            genders = [d["gender"] for d in cl["detections"]]
            ages = [d["age"] for d in cl["detections"]]
            female_count = genders.count("female")
            male_count = genders.count("male")
            dominant_gender = "female" if female_count >= male_count else "male"
            avg_age = int(round(np.mean(ages))) if ages else 25

            results.append({
                "cluster_id": cl["cluster_id"],
                "gender": dominant_gender,
                "age": avg_age,
                "embedding": cl["centroid_embedding"],
                "detection_count": len(cl["detections"]),
            })

        # Sort by frequency of appearance (most prominent actors first)
        results.sort(key=lambda x: x["detection_count"], reverse=True)
        return results

    def extract_face_from_image(self, image_path: Path):
        """Detect and return the main face object from an AI generated reference image."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load character image: {image_path}")

        # Try default detection
        faces = self.face_app.get(img)

        # If no face found with default (640x640), try multi-scale detection
        if not faces:
            h, w = img.shape[:2]
            for det_sz in [(w, h), (1024, 1024), (512, 512), (320, 320)]:
                try:
                    self.face_app.prepare(ctx_id=0, det_size=det_sz)
                    faces = self.face_app.get(img)
                    if faces:
                        break
                except Exception:
                    pass
            # Reset detector to standard size
            try:
                self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            except Exception:
                pass

        if not faces:
            raise ValueError(f"No face detected in reference image: {image_path}")
        # Choose the largest face in reference image
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return faces[0]

    @staticmethod
    def estimate_face_yaw(face) -> str:
        """
        Estimate if face is front-facing or turned (3/4 side angle)
        using 5-point facial landmarks (kps: left_eye, right_eye, nose, left_mouth, right_mouth).
        """
        if not hasattr(face, "kps") or face.kps is None or len(face.kps) < 5:
            return "front"
        kps = face.kps
        left_eye, right_eye, nose = kps[0], kps[1], kps[2]
        eye_width = abs(right_eye[0] - left_eye[0])
        if eye_width < 1e-4:
            return "side"
        # Ratio of nose position between eyes: ~0.5 means looking straight ahead
        nose_ratio = (nose[0] - left_eye[0]) / eye_width
        if nose_ratio < 0.35 or nose_ratio > 0.65:
            return "side"
        return "front"

    def swap_video_faces(
        self,
        source_video: Path,
        mapping: Dict[int, Any],  # cluster_id -> dict of {"front": face, "side": face} or single face
        cluster_centroids: Dict[int, np.ndarray],
        output_video: Path,
    ) -> Path:
        """
        Replaces detected faces frame-by-frame according to the cluster-to-face mapping,
        preserving audio and timing.
        """
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(source_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError(f"Could not read valid dimensions from {source_video} ({width}x{height})")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tempfile.TemporaryDirectory(prefix="cloning_swap_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_raw_video = temp_dir_path / "swapped_raw.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(temp_raw_video), fourcc, fps, (width, height))

            logger.info(f"[CLONING] Swapping faces across {total_frames} frames at {fps} FPS...")
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                faces = self.face_app.get(frame)
                for face in faces:
                    emb = face.normed_embedding if hasattr(face, "normed_embedding") else face.embedding / np.linalg.norm(face.embedding)
                    
                    # Match detected face to known cluster centroids
                    best_cluster_id = None
                    best_sim = -1.0
                    for c_id, centroid in cluster_centroids.items():
                        sim = float(np.dot(centroid, emb))
                        if sim > best_sim:
                            best_sim = sim
                            best_cluster_id = c_id

                    # Only swap if similarity is confident and we have a target face for this cluster
                    if best_cluster_id is not None and best_sim >= 0.38 and best_cluster_id in mapping:
                        target_entry = mapping[best_cluster_id]
                        if isinstance(target_entry, dict):
                            # Dynamically select front vs 3/4 side angle based on actor's face orientation
                            detected_pose = self.estimate_face_yaw(face)
                            target_face = target_entry.get(detected_pose) or target_entry.get("front") or next(iter(target_entry.values()))
                        else:
                            target_face = target_entry

                        frame = self.swapper.get(frame, face, target_face, paste_back=True)

                writer.write(frame)
                frame_idx += 1
                if frame_idx % 50 == 0:
                    logger.debug(f"[CLONING] Processed {frame_idx}/{total_frames} frames")

            cap.release()
            writer.release()

            # Re-stitch audio using ffmpeg
            logger.info("[CLONING] Finalizing video assembly and preserving audio track...")
            cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_raw_video),
                "-i", str(source_video),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                str(output_video)
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                # If audio copy failed (e.g. video has no audio track), fallback to video only
                cmd_no_audio = [
                    "ffmpeg", "-y",
                    "-i", str(temp_raw_video),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    str(output_video)
                ]
                subprocess.run(cmd_no_audio, check=True)

        logger.info(f"[CLONING] Completed face replacement -> {output_video}")
        return output_video

