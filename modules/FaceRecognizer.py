"""
Face Recognition Module using InsightFace (RetinaFace detector + ArcFace embeddings)
No external YOLO face model is needed — InsightFace handles both detection
and recognition in a single call.
"""

import numpy as np
import cv2
from insightface.app import FaceAnalysis
from typing import List, Dict, Optional, Tuple
import logging
import pickle
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Face detection + recognition using InsightFace.
    RetinaFace (built-in) detects faces, ArcFace produces embeddings.
    """

    def __init__(self, config: Dict):
        self.config = config['recognition']
        self.embedding_size = self.config['embedding_size']
        self.similarity_threshold = self.config['similarity_threshold']
        use_gpu = self.config.get('use_gpu', False)

        det_size = tuple(self.config.get('det_size', [640, 640]))

        try:
            providers = (
                ['CUDAExecutionProvider', 'CPUExecutionProvider']
                if use_gpu else ['CPUExecutionProvider']
            )
            self.face_app = FaceAnalysis(
                name=self.config['model'],
                root='./models',
                providers=providers,
            )
            self.face_app.prepare(
                ctx_id=0 if use_gpu else -1,
                det_size=det_size,
            )
            logger.info(f"InsightFace loaded: {self.config['model']}  det_size={det_size}")
        except Exception as e:
            logger.error(f"Failed to load InsightFace: {e}")
            raise

        # Registry
        self.registered_faces: Dict[str, np.ndarray] = {}
        self.face_metadata: Dict[str, Dict] = {}

        logger.info(f"FaceRecognizer initialized  threshold={self.similarity_threshold}")

    # ------------------------------------------------------------------
    # Combined detect + embed  (replaces generate_embedding)
    # ------------------------------------------------------------------

    def detect_and_embed(self, image: np.ndarray) -> List[Dict]:
        """
        Run InsightFace on an image: detect all faces and return embeddings.

        Args:
            image: BGR image (full body crop or full frame)

        Returns:
            List of dicts:
              {
                'embedding': np.ndarray (512-d),
                'bbox': [x1, y1, x2, y2],           # face bbox within input image
                'confidence': float,
                'face_crop': np.ndarray | None,      # tight face crop
              }
        """
        if image is None or image.size == 0:
            return []

        try:
            # Pad the image to help RetinaFace detect faces in tight body crops
            h, w = image.shape[:2]
            pad_h, pad_w = int(h * 0.3), int(w * 0.3)
            padded = cv2.copyMakeBorder(
                image, pad_h, pad_h, pad_w, pad_w,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

            faces = self.face_app.get(padded)

            if not faces:
                # Retry without padding
                faces = self.face_app.get(image)
                pad_h, pad_w = 0, 0

            results = []
            for face in faces:
                emb = face.normed_embedding
                if emb is None:
                    continue

                # Map bbox back from padded coords
                fb = face.bbox.astype(int)
                x1 = max(0, fb[0] - pad_w)
                y1 = max(0, fb[1] - pad_h)
                x2 = min(w, fb[2] - pad_w)
                y2 = min(h, fb[3] - pad_h)

                face_crop = image[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

                results.append({
                    'embedding': emb,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(face.det_score),
                    'face_crop': face_crop,
                })

            return results

        except Exception as e:
            logger.error(f"detect_and_embed error: {e}")
            return []

    # ------------------------------------------------------------------
    # Legacy generate_embedding (kept for backward compat)
    # ------------------------------------------------------------------

    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Generate embedding from a face/body crop using InsightFace."""
        results = self.detect_and_embed(face_image)
        if results:
            return results[0]['embedding']
        return None

    # ------------------------------------------------------------------
    # Registration & matching
    # ------------------------------------------------------------------

    def register_face(self, face_id: str, embedding: np.ndarray, metadata: Dict = None) -> bool:
        try:
            self.registered_faces[face_id] = embedding
            self.face_metadata[face_id] = metadata or {}
            logger.info(f"Registered new face with ID: {face_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register face {face_id}: {e}")
            return False

    def recognize_face(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Compare embedding against registry. Returns (face_id, similarity) or (None, 0)."""
        if embedding is None or not self.registered_faces:
            return None, 0.0

        best_id = None
        best_sim = 0.0

        for fid, reg_emb in self.registered_faces.items():
            # Ensure we are comparing 1-D arrays
            dist = cosine(embedding.flatten(), reg_emb.flatten())
            sim = 1.0 - float(dist)
            if sim > best_sim:
                best_sim = sim
                best_id = fid

        if best_sim >= self.similarity_threshold:
            return best_id, best_sim
        return None, best_sim

    def verify_faces(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return 1.0 - cosine(emb1, emb2)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_registry(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'embeddings': self.registered_faces, 'metadata': self.face_metadata}, f)
        logger.info(f"Saved face registry to {filepath}")

    def load_registry(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.registered_faces = data['embeddings']
            self.face_metadata = data['metadata']
            logger.info(f"Loaded {len(self.registered_faces)} faces from {filepath}")
        except FileNotFoundError:
            logger.warning(f"Registry file not found: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")