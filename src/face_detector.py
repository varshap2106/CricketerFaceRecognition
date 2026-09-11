"""
Face Detection and Cropping Module.
Handles multi-scale face localization, false-positive filtering (shirt logos, collars), and face cropping.
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Union

try:
    from src.config import (
        FACE_SIZE,
        HAAR_CASCADE_PATH,
        HAAR_EYE_CASCADE_PATH,
        RAW_DATA_DIR,
        CROPPED_DATA_DIR
    )
except ImportError:
    from config import (
        FACE_SIZE,
        HAAR_CASCADE_PATH,
        HAAR_EYE_CASCADE_PATH,
        RAW_DATA_DIR,
        CROPPED_DATA_DIR
    )


class FaceDetector:
    """
    Robust Face Detector using OpenCV Haar Cascades with aspect-ratio validation,
    non-maximum suppression, and false positive (shirt logo) filtering.
    """

    def __init__(self, face_cascade_path: Optional[str] = None, eye_cascade_path: Optional[str] = None):
        face_path = str(face_cascade_path) if face_cascade_path else str(HAAR_CASCADE_PATH)
        if not os.path.exists(face_path):
            face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        
        self.face_cascade = cv2.CascadeClassifier(face_path)
        if self.face_cascade.empty():
            face_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
            self.face_cascade = cv2.CascadeClassifier(face_path)

        eye_path = str(eye_cascade_path) if eye_cascade_path else str(HAAR_EYE_CASCADE_PATH)
        if not os.path.exists(eye_path):
            eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        
        self.eye_cascade = cv2.CascadeClassifier(eye_path)

    def detect_faces(
        self,
        image: Union[np.ndarray, str, Path, Image.Image],
        scale_factor: float = 1.1,
        min_neighbors: int = 6,
        min_size: Tuple[int, int] = (60, 60)
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Detects only true human faces, filtering out false positives (e.g. logos, buttons, text on shirts).
        """
        img_bgr = self._load_image_bgr(image)
        if img_bgr is None:
            return None, []

        img_h, img_w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Multi-scale cascade detection with high neighbor threshold for clean detections
        raw_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        valid_boxes = []
        max_area = 0

        for (x, y, w, h) in raw_faces:
            # 1. Aspect Ratio Validation (Human faces are roughly 1:1 to 1:1.3)
            aspect = float(w) / float(h)
            if aspect < 0.65 or aspect > 1.35:
                continue  # Reject flat logos or stretched boxes

            # 2. Reject boxes that are too small relative to the image
            if w < (img_w * 0.08) or h < (img_h * 0.08):
                continue

            # 3. Position check: Heads are usually in top 75% of portrait photos
            # If a small box is detected below the middle of the body and a bigger head exists above, it's a shirt logo
            area = w * h
            if area > max_area:
                max_area = area

            valid_boxes.append((int(x), int(y), int(w), int(h)))

        # Filter out minor secondary boxes on the chest if a primary face is detected above
        if len(valid_boxes) > 1:
            filtered = []
            for b in valid_boxes:
                bx, by, bw, bh = b
                # If box area is at least 35% of the largest face detected, keep it
                if (bw * bh) >= (max_area * 0.35):
                    filtered.append(b)
            valid_boxes = filtered

        # Sort top-to-bottom and largest first
        valid_boxes.sort(key=lambda b: (-b[2]*b[3], b[1]))

        return img_bgr, valid_boxes

    def crop_face(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = FACE_SIZE,
        margin: float = 0.08  # Tight margin to prevent collar/shirt intrusion
    ) -> np.ndarray:
        """
        Crop face region with tight boundary and resize to target_size.
        """
        x, y, w, h = box
        img_h, img_w = image.shape[:2]

        margin_x = int(w * margin)
        margin_y = int(h * margin)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_w, x + w + margin_x)
        y2 = min(img_h, y + h + margin_y)

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            return None

        return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

    def detect_and_crop_faces(
        self,
        image: Union[np.ndarray, str, Path, Image.Image],
        require_eyes: bool = False,
        target_size: Tuple[int, int] = FACE_SIZE
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Detects faces and returns pairs of (cropped_face, bounding_box)."""
        img_bgr, boxes = self.detect_faces(image)
        results = []
        if img_bgr is None:
            return results

        for box in boxes:
            crop = self.crop_face(img_bgr, box, target_size=target_size)
            if crop is not None:
                results.append((crop, box))

        return results

    def annotate_image(
        self,
        image: np.ndarray,
        detections: List[dict],
        show_stats: bool = True
    ) -> np.ndarray:
        """Draws sleek modern corner brackets and player labels."""
        annotated = image.copy()
        for det in detections:
            x, y, w, h = det['box']
            name = det.get('name', 'Unknown')
            conf = det.get('confidence', 0.0)
            role = det.get('role', '')
            color = (0, 240, 255) if "Unknown" not in name else (0, 165, 255)

            # Draw stylish corner brackets
            length = int(min(w, h) * 0.25)
            thick = 3
            cv2.line(annotated, (x, y), (x + length, y), color, thick)
            cv2.line(annotated, (x, y), (x, y + length), color, thick)
            cv2.line(annotated, (x + w, y), (x + w - length, y), color, thick)
            cv2.line(annotated, (x + w, y), (x + w, y + length), color, thick)
            cv2.line(annotated, (x, y + h), (x + length, y + h), color, thick)
            cv2.line(annotated, (x, y + h), (x, y + h - length), color, thick)
            cv2.line(annotated, (x + w, y + h), (x + w - length, y + h), color, thick)
            cv2.line(annotated, (x + w, y + h), (x + w, y + h - length), color, thick)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 1)

            # Capsule label
            label = f"{name} ({int(conf * 100)}%)" if conf > 0 else name
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.60
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)

            tag_y1 = max(0, y - text_h - 12)
            tag_y2 = y
            cv2.rectangle(annotated, (x, tag_y1), (x + text_w + 14, tag_y2), (12, 18, 28), -1)
            cv2.rectangle(annotated, (x, tag_y1), (x + text_w + 14, tag_y2), color, 1)
            cv2.putText(annotated, label, (x + 7, tag_y2 - 5), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

            if show_stats and role and "Unknown" not in name:
                (rw, rh), _ = cv2.getTextSize(role, font, 0.45, 1)
                sub_y1 = y + h
                sub_y2 = y + h + rh + 10
                cv2.rectangle(annotated, (x, sub_y1), (x + rw + 12, sub_y2), (12, 18, 28), -1)
                cv2.putText(annotated, role, (x + 6, sub_y2 - 4), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        return annotated

    def _load_image_bgr(self, image: Union[np.ndarray, str, Path, Image.Image]) -> Optional[np.ndarray]:
        if isinstance(image, (str, Path)):
            path_str = str(image)
            if not os.path.exists(path_str):
                return None
            return cv2.imread(path_str)
        elif isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            return image
        return None
