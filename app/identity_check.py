"""
identity_check.py - face-identity verification + retry for generated keyframes.

Requires:  pip install insightface onnxruntime-gpu opencv-python numpy
(use onnxruntime instead of onnxruntime-gpu if you want CPU only)

Typical use inside SceneManager.generate_keyframes:

    checker = IdentityChecker(threshold=0.38)          # create once
    checker.register(char_id, reference_sheet_path)    # once per character

    result = checker.generate_with_retry(
        generate_fn=lambda seed: self._make_keyframe(seed),   # returns Path
        expected_ids=scene_character_ids,
        base_seed=2000003 + order * 7919,
        max_tries=3,
    )
"""
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class IdentityChecker:
    def __init__(self, threshold: float = 0.38, front_panel_fraction: float = 1 / 3,
                 det_size=(640, 640)):
        # ArcFace cosine similarity: same person is usually > 0.4.
        # AI-generated faces drift, so 0.35-0.40 is a practical gate. Tune on your data.
        from insightface.app import FaceAnalysis
        try:
            import onnxruntime as ort
            avail = ort.get_available_providers()
        except Exception:
            avail = []
        use_gpu = "CUDAExecutionProvider" in avail
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu
                     else ["CPUExecutionProvider"])
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size)
        self.threshold = threshold
        self.front_fraction = front_panel_fraction
        self.ref_embeddings: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def register(self, character_id: str, reference_sheet: str):
        """Embed the FRONT panel (left third) of the 3-panel turnaround sheet."""
        img = cv2.imread(str(reference_sheet))
        if img is None:
            raise FileNotFoundError(reference_sheet)
        w = img.shape[1]
        front = img[:, : int(w * self.front_fraction)]
        faces = self.app.get(front) or self.app.get(img)   # fall back to whole sheet
        if not faces:
            logger.warning(f"[IDENTITY] No face found in reference for {character_id}; "
                           "identity check disabled for this character.")
            return
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        self.ref_embeddings[character_id] = face.normed_embedding

    # ------------------------------------------------------------------
    def check(self, image_path: str, expected_ids: List[str]) -> Dict:
        """
        Returns {"ok": bool, "scores": {cid: sim|None}, "failures": [str]}
        - A character with no detectable face (back view, cropped) is skipped, not failed.
        - Each detected face may be claimed by only one character, so two people
          who drift into the same face are caught.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return {"ok": False, "scores": {}, "failures": [f"cannot read {image_path}"]}
        faces = self.app.get(img)
        scores: Dict[str, Optional[float]] = {}
        failures: List[str] = []

        ids = [c for c in expected_ids if c in self.ref_embeddings]
        if not faces:
            return {"ok": True, "scores": {c: None for c in ids}, "failures": [],
                    "note": "no faces detected (rear/cropped shot) - skipped"}

        # similarity matrix: characters x faces
        sim = np.array([[float(np.dot(self.ref_embeddings[c], f.normed_embedding))
                         for f in faces] for c in ids]) if ids else np.zeros((0, 0))

        # greedy one-to-one assignment, strongest match first
        used_faces, assigned = set(), {}
        pairs = sorted(((sim[i, j], i, j) for i in range(len(ids)) for j in range(len(faces))),
                       reverse=True)
        for s, i, j in pairs:
            if ids[i] in assigned or j in used_faces:
                continue
            assigned[ids[i]] = float(s)
            used_faces.add(j)

        for cid in ids:
            s = assigned.get(cid)
            scores[cid] = s
            if s is None:
                continue  # fewer faces than characters: someone is not facing camera
            if s < self.threshold:
                failures.append(f"{cid}: similarity {s:.2f} < {self.threshold}")

        return {"ok": not failures, "scores": scores, "failures": failures}

    # ------------------------------------------------------------------
    def generate_with_retry(self, generate_fn: Callable[[int], Path],
                            expected_ids: List[str], base_seed: int,
                            max_tries: int = 3) -> Dict:
        """
        generate_fn(seed) must produce the keyframe and return its path.
        Keeps the best attempt (highest minimum similarity) if all tries fail.
        """
        if max_tries < 1:
            raise ValueError("max_tries must be at least 1")
        best = None
        for attempt in range(max_tries):
            seed = base_seed + attempt * 104729
            path = generate_fn(seed)
            report = self.check(str(path), expected_ids)
            valid = [s for s in report["scores"].values() if s is not None]
            quality = min(valid) if valid else 1.0
            logger.info(f"[IDENTITY] try {attempt + 1}/{max_tries} seed={seed} "
                        f"scores={report['scores']} ok={report['ok']}")
            if best is None or quality > best["quality"]:
                best = {"path": path, "seed": seed, "report": report, "quality": quality}
            if report["ok"]:
                break
        best["passed"] = best["report"]["ok"]
        best["identity_warning"] = not best["passed"]   # store in scene metadata
        if not best["passed"]:
            logger.warning(f"[IDENTITY] all {max_tries} tries below threshold; "
                           f"using best (min sim {best['quality']:.2f}).")
        return best
