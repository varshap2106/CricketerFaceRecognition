"""
Feature Extraction and Image Preprocessing Module.
Integrates Deep Learning 128-D Metric Embeddings with multi-scale HOG and CLAHE lighting invariance.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional

try:
    from src.face_recognition_dl import DeepFaceRecognizer
    from src.advanced_features import HighAccuracyFeatureExtractor
    from src.config import FACE_SIZE
except ImportError:
    from face_recognition_dl import DeepFaceRecognizer
    from advanced_features import HighAccuracyFeatureExtractor
    from config import FACE_SIZE


class FeatureExtractor:
    """
    Unified High-Performance Feature Extractor combining Deep Learning embeddings
    with geometric texture descriptors.
    """

    def __init__(self, face_size: Tuple[int, int] = FACE_SIZE):
        self.face_size = face_size
        self.deep_engine = DeepFaceRecognizer()
        self.fallback_engine = HighAccuracyFeatureExtractor()

    def extract_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts 128-D deep metric embedding or multi-scale normalized embedding.
        """
        if face_bgr is None or face_bgr.size == 0:
            return None

        # 1. Primary: Deep metric 128-D embedding
        deep_emb = self.deep_engine.extract_deep_embedding(face_bgr)
        if deep_emb is not None and len(deep_emb) > 0:
            return deep_emb

        # 2. Secondary fallback: High-accuracy dense descriptor
        return self.fallback_engine.extract_embedding(face_bgr)

    def extract_combined_features(self, face_bgr: np.ndarray) -> np.ndarray:
        return self.extract_embedding(face_bgr)


class DataAugmentor:
    """Augments face images to increase dataset variety, reducing overfitting."""

    @staticmethod
    def augment_image(img_bgr: np.ndarray) -> List[np.ndarray]:
        if img_bgr is None:
            return []

        augmented = [img_bgr]

        # 1. Flip
        flipped = cv2.flip(img_bgr, 1)
        augmented.append(flipped)

        # 2. Brightness adjustment
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        bright = cv2.cvtColor(cv2.merge((h, s, cv2.add(v, 25))), cv2.COLOR_HSV2BGR)
        dark = cv2.cvtColor(cv2.merge((h, s, cv2.subtract(v, 20))), cv2.COLOR_HSV2BGR)
        augmented.extend([bright, dark])

        # 3. Slight Rotation
        h_img, w_img = img_bgr.shape[:2]
        center = (w_img // 2, h_img // 2)
        for angle in [-6, 6]:
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img_bgr, rot_mat, (w_img, h_img), borderMode=cv2.BORDER_REFLECT)
            augmented.append(rotated)

        return augmented
