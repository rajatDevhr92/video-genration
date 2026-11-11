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

INSWAPPER_MIRRORS = [
    "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
    "https://huggingface.co/ModelsLab/inswapper/resolve/main/inswapper_128.onnx",
]
MODELS_DIR = Path(os.getenv("MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))


def find_or_download_inswapper() -> Path:
    """Finds existing inswapper_128.onnx across workspace, ComfyUI, and user cache, or downloads it."""
    candidate_locations = [
        MODELS_DIR / "inswapper_128.onnx",
        Path("/workspace/story-video-engine/models/inswapper_128.onnx"),
        Path("/workspace/ComfyUI/models/insightface/inswapper_128.onnx"),
        Path("/workspace/ComfyUI/models/reactor/faceswap/inswapper_128.onnx"),
        Path("/workspace/ComfyUI/models/reactor/inswapper_128.onnx"),
        Path("/root/.insightface/models/inswapper_128.onnx"),
        Path.home() / ".insightface" / "models" / "inswapper_128.onnx",
    ]
    for p in candidate_locations:
        if p.exists() and p.stat().st_size > 500_000_000:
            logger.info(f"Using existing InSwapper model at {p} ({p.stat().st_size / (1024*1024):.1f} MB)")
            return p

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = MODELS_DIR / "inswapper_128.onnx"

    if target_path.exists() and target_path.stat().st_size < 500_000_000:
        logger.warning(f"Incomplete {target_path} ({target_path.stat().st_size} bytes). Removing to re-download.")
        target_path.unlink(missing_ok=True)

    if target_path.exists() and target_path.stat().st_size >= 500_000_000:
        return target_path

    # Try downloading with curl first (most reliable on Linux / Runpod / GPU instances)
    logger.info(f"Downloading InSwapper model (~528 MB) to {target_path}...")
    for url in INSWAPPER_MIRRORS:
        try:
            logger.info(f"Downloading InSwapper from {url} via curl...")
            subprocess.run(
                ["curl", "-L", "-k", "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "-o", str(target_path), url],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600,
            )
            if target_path.exists() and target_path.stat().st_size > 500_000_000:
                logger.info(f"InSwapper model successfully downloaded to {target_path} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
                return target_path
        except Exception as curl_err:
            logger.warning(f"curl download failed: {curl_err}")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=600) as resp, open(str(target_path), "wb") as f_out:
                shutil.copyfileobj(resp, f_out)
            if target_path.exists() and target_path.stat().st_size > 500_000_000:
                logger.info(f"InSwapper model successfully downloaded via urllib to {target_path}")
                return target_path
        except Exception as url_err:
            logger.warning(f"urllib download failed: {url_err}")

    if not target_path.exists() or target_path.stat().st_size < 500_000_000:
        target_path.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"InSwapper model (inswapper_128.onnx, ~528MB) is required for face swapping but could not be downloaded.\n"
            f"Run this command on your server terminal to install it:\n"
            f"mkdir -p /workspace/story-video-engine/models && "
            f"wget -O /workspace/story-video-engine/models/inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
        )
    return target_path


def get_inswapper_path() -> Path:
    """Ensure the inswapper_128.onnx model is downloaded and return its path."""
    return find_or_download_inswapper()


class FaceSwapGenerator:
    def __init__(self, swapper_model_path: Optional[Path] = None):
        self.swapper_model_path = swapper_model_path or get_inswapper_path()
        self._face_app = None
        self._swapper = None

    @property
    def face_app(self):
        if self._face_app is None:
            from insightface.app import FaceAnalysis
            import onnxruntime as ort
            providers = ort.get_available_providers()
            active_provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
            print(f"[DEVICE LOG] InsightFace FaceAnalysis running on: >>> {active_provider} <<< (Available: {providers})")
            logger.info(f"[DEVICE LOG] InsightFace FaceAnalysis running on: {active_provider}")
            app = FaceAnalysis(name="buffalo_l", providers=[active_provider, "CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.25)
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
            import onnxruntime as ort
            providers = ort.get_available_providers()
            active_provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
            print(f"[DEVICE LOG] InSwapper model running on: >>> {active_provider} <<< (Available: {providers})")
            logger.info(f"[DEVICE LOG] InSwapper model running on: {active_provider}")
            self._swapper = insightface.model_zoo.get_model(
                str(self.swapper_model_path),
                download=False,
                providers=[active_provider, "CPUExecutionProvider"]
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

    def _resolve_face_obj(self, item: Any, preferred_pose: Optional[str] = None):
        """Converts a Path, str, dict of angles, or Face into a valid insightface Face object."""
        if item is None:
            return None
        if isinstance(item, (str, Path)):
            item_path = Path(item)
            if item_path.exists():
                try:
                    return self.extract_face_from_image(item_path)
                except Exception as e:
                    logger.warning(f"Could not extract face from {item_path}: {e}")
            return None
        if isinstance(item, dict):
            # Check for angle keys
            if preferred_pose and preferred_pose in item:
                val = item[preferred_pose]
            elif "front" in item:
                val = item["front"]
            elif len(item) > 0:
                val = next(iter(item.values()))
            else:
                return None
            return self._resolve_face_obj(val, preferred_pose=preferred_pose)
        # Item is already a Face object
        return item

    def detect_faces_robust(self, img: np.ndarray, min_faces: int = 1) -> List[Any]:
        """
        Detects faces in img. If fewer than min_faces are detected, tries:
        1. Higher resolution detection (1280x1280, det_thresh=0.15)
        2. Image rotation (90 deg counter-clockwise and clockwise) to catch angled/downward faces,
           mapping coordinates and landmarks back to original image space.
        """
        if img is None:
            return []

        # 1. Standard detection
        faces = self.face_app.get(img)
        if len(faces) >= min_faces:
            return sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

        # 2. Try HD multi-scale detection
        h, w = img.shape[:2]
        try:
            self.face_app.prepare(ctx_id=0, det_size=(1280, 1280), det_thresh=0.15)
            hd_faces = self.face_app.get(img)
            if len(hd_faces) > len(faces):
                faces = hd_faces
        except Exception:
            pass
        finally:
            self.face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.20)

        if len(faces) >= min_faces:
            return sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

        # 3. Multi-rotation detection for bent-over / downward action scenes
        rotations = [
            (cv2.ROTATE_90_COUNTERCLOCKWISE, "ccw"),
            (cv2.ROTATE_90_CLOCKWISE, "cw"),
        ]
        for rot_code, rot_name in rotations:
            rot_img = cv2.rotate(img, rot_code)
            rot_faces = self.face_app.get(rot_img)
            for rf in rot_faces:
                orig_bbox = np.zeros_like(rf.bbox)
                orig_kps = np.zeros_like(rf.kps)
                if rot_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
                    orig_bbox[0] = w - 1 - rf.bbox[3]
                    orig_bbox[1] = rf.bbox[0]
                    orig_bbox[2] = w - 1 - rf.bbox[1]
                    orig_bbox[3] = rf.bbox[2]
                    for ki in range(len(rf.kps)):
                        orig_kps[ki][0] = w - 1 - rf.kps[ki][1]
                        orig_kps[ki][1] = rf.kps[ki][0]
                elif rot_code == cv2.ROTATE_90_CLOCKWISE:
                    orig_bbox[0] = rf.bbox[1]
                    orig_bbox[1] = h - 1 - rf.bbox[2]
                    orig_bbox[2] = rf.bbox[3]
                    orig_bbox[3] = h - 1 - rf.bbox[0]
                    for ki in range(len(rf.kps)):
                        orig_kps[ki][0] = rf.kps[ki][1]
                        orig_kps[ki][1] = h - 1 - rf.kps[ki][0]

                # Check if duplicate by IoU
                is_duplicate = False
                for existing in faces:
                    x1 = max(existing.bbox[0], orig_bbox[0])
                    y1 = max(existing.bbox[1], orig_bbox[1])
                    x2 = min(existing.bbox[2], orig_bbox[2])
                    y2 = min(existing.bbox[3], orig_bbox[3])
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    existing_area = (existing.bbox[2] - existing.bbox[0]) * (existing.bbox[3] - existing.bbox[1])
                    if existing_area > 0 and inter > 0.3 * existing_area:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    rf.bbox = orig_bbox
                    rf.kps = orig_kps
                    faces.append(rf)

            if len(faces) >= min_faces:
                break

        return sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

    def swap_face_in_image(self, image_path: Path, target_faces: Any, output_path: Path) -> Path:
        """
        Swaps target_faces into the detected actors in image_path,
        preserving the original room, bodies, clothing, and background.
        target_faces can be:
          - A single Face object or Path to face image
          - A dict mapping gender ('female', 'male') or cluster_id to Face object / Path / angle dict
          - A list of Face objects / Paths
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read scene image file at {image_path}")

        faces = self.detect_faces_robust(img, min_faces=2)
        if not faces:
            raise RuntimeError(
                f"No face detected in scene frame {image_path.name}. "
                "Ensure the scene frame contains at least one visible actor face."
            )

        # Sort largest face first
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        swapped_count = 0

        for idx, face in enumerate(faces):
            g_raw = getattr(face, "gender", None)
            if g_raw is None:
                g_raw = 0 if getattr(face, "sex", "F") == "F" else 1
            actor_gender = "female" if g_raw == 0 else "male"
            pose = self.estimate_face_yaw(face)

            candidate = None
            if isinstance(target_faces, dict):
                # Try exact gender match ("female" or "male")
                candidate = target_faces.get(actor_gender)
                if candidate is None:
                    candidate = target_faces.get(0 if actor_gender == "female" else 1)
                if candidate is None:
                    candidate = target_faces.get(idx)
                if candidate is None and len(target_faces) > 0:
                    candidate = next(iter(target_faces.values()))
            elif isinstance(target_faces, (list, tuple)):
                candidate = target_faces[min(idx, len(target_faces) - 1)]
            else:
                candidate = target_faces

            face_obj = self._resolve_face_obj(candidate, preferred_pose=pose)
            if face_obj is not None:
                try:
                    img = self.swapper.get(img, face, face_obj, paste_back=True)
                    swapped_count += 1
                    logger.info(f"[FACE SWAP] Successfully swapped actor {idx} ({actor_gender}) in {image_path.name}")
                    print(f"[FACE SWAP] Successfully swapped actor {idx} ({actor_gender}) in {image_path.name}")
                except Exception as swap_err:
                    logger.error(f"[FACE SWAP ERROR] Could not swap face {idx} ({actor_gender}): {swap_err}", exc_info=True)
                    print(f"[FACE SWAP ERROR] Could not swap face {idx} ({actor_gender}): {swap_err}")
                    raise

        if swapped_count == 0:
            raise RuntimeError(
                f"Face swap failed to replace any faces in {image_path.name}. "
                f"Detected {len(faces)} actor face(s), but swapper failed. "
                "Verify that inswapper_128.onnx (~528MB) is installed in models/."
            )

        cv2.imwrite(str(output_path), img)
        return output_path

    def swap_video_faces(
        self,
        source_video: Path,
        mapping: Dict[int, Any],  # cluster_id -> dict of {"front": face, "side": face} or single face
        cluster_centroids: Dict[int, np.ndarray],
        output_video: Path,
        cluster_genders: Optional[Dict[int, str]] = None,
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
            print(f"[CLONING] Swapping faces across {total_frames} frames at {fps} FPS...")
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

                    # If similarity is low or angled profile, match by detected gender as reliable fallback
                    if (best_sim < 0.28 or best_cluster_id not in mapping) and cluster_genders:
                        detected_gender = "female" if getattr(face, "gender", 0) == 0 else "male"
                        for c_id, g_name in cluster_genders.items():
                            if g_name.lower() == detected_gender and c_id in mapping:
                                best_cluster_id = c_id
                                best_sim = 1.0
                                break

                    threshold = 0.20 if len(cluster_centroids) <= 1 else 0.28
                    if best_cluster_id is not None and best_sim >= threshold and best_cluster_id in mapping:
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
                if frame_idx % 25 == 0 or frame_idx == total_frames:
                    print(f"[CLONING PROGRESS] Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")
                    logger.info(f"[CLONING PROGRESS] Processed {frame_idx}/{total_frames} frames")

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

