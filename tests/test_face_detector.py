"""
Unit tests for FaceDetector and Preprocessor modules.
"""

import unittest
import numpy as np
import cv2
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.face_detector import FaceDetector
from src.preprocessor import FeatureExtractor, DataAugmentor
from src.config import FACE_SIZE, RAW_PIXEL_RESIZE


class TestFaceDetectorAndPreprocessor(unittest.TestCase):

    def setUp(self):
        self.detector = FaceDetector()
        self.extractor = FeatureExtractor()
        
        # Create a synthetic face test image
        self.test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        self.test_img[:] = (20, 20, 30)
        # Face ellipse
        cv2.ellipse(self.test_img, (150, 150), (60, 80), 0, 0, 360, (140, 170, 210), -1)
        # Eyes
        cv2.circle(self.test_img, (130, 130), 8, (255, 255, 255), -1)
        cv2.circle(self.test_img, (170, 130), 8, (255, 255, 255), -1)

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector.face_cascade)

    def test_crop_face_dimensions(self):
        box = (100, 80, 100, 140)
        crop = self.detector.crop_face(self.test_img, box, target_size=FACE_SIZE)
        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape[:2], (FACE_SIZE[1], FACE_SIZE[0]))

    def test_feature_extraction(self):
        face_crop = cv2.resize(self.test_img, FACE_SIZE)
        feat = self.extractor.extract_combined_features(face_crop)
        self.assertIsNotNone(feat)
        # Expected feature dimension: (64*64*3) + (64*64*1) = 12288 + 4096 = 16384
        expected_len = (RAW_PIXEL_RESIZE[0] * RAW_PIXEL_RESIZE[1] * 3) + (RAW_PIXEL_RESIZE[0] * RAW_PIXEL_RESIZE[1] * 1)
        self.assertEqual(len(feat), expected_len)

    def test_data_augmentor(self):
        face_crop = cv2.resize(self.test_img, FACE_SIZE)
        augmented = DataAugmentor.augment_image(face_crop)
        self.assertGreaterEqual(len(augmented), 3)


if __name__ == "__main__":
    unittest.main()
