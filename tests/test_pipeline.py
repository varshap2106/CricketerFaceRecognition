"""
Integration tests for dataset generation, model training, evaluation, and inference pipeline.
"""

import unittest
import numpy as np
import cv2
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.dataset_downloader import DatasetManager
from src.train import ModelTrainer
from src.predict import FacePredictor
from src.evaluate import ModelEvaluator
from src.config import MODEL_PATH, CLASS_DICT_PATH, CROPPED_DATA_DIR


class TestEndToEndPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate minimal fast dataset for quick integration test
        dm = DatasetManager()
        dm.generate_starter_dataset(samples_per_player=10)

    def test_model_training_and_serialization(self):
        trainer = ModelTrainer()
        results = trainer.train_and_tune(test_size=0.20)
        self.assertTrue(MODEL_PATH.exists())
        self.assertTrue(CLASS_DICT_PATH.exists())
        self.assertGreater(results["best_accuracy"], 0.50)

    def test_predictor_inference(self):
        predictor = FacePredictor()
        self.assertTrue(predictor.is_model_ready())

        # Test on an existing sample
        sample_path = list(CROPPED_DATA_DIR.glob("*/*.jpg"))[0]
        results = predictor.predict_image(sample_path)

        self.assertEqual(results["status"], "success")
        self.assertGreaterEqual(len(results["detections"]), 1)
        self.assertIn("player_name", results["detections"][0])
        self.assertIn("confidence", results["detections"][0])

    def test_evaluator_metrics(self):
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model()
        self.assertIn("overall_accuracy", metrics)
        self.assertGreater(metrics["overall_accuracy"], 0.50)


if __name__ == "__main__":
    unittest.main()
