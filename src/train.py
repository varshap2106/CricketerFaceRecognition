"""
Model Training and Hyperparameter Tuning Module.
Extracts multi-channel facial features (Wavelet, HOG, LBP, pixel intensities),
performs PCA dimensionality reduction, computes class centroids, and trains classifiers.
"""

import os
import json
import joblib
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, List, Any
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

try:
    from src.config import (
        CROPPED_DATA_DIR,
        MODEL_PATH,
        SCALER_PCA_PATH,
        CLASS_DICT_PATH,
        INDIAN_CRICKETERS,
        PLAYER_DISPLAY_NAMES
    )
    from src.preprocessor import FeatureExtractor
    from src.fetch_dataset import download_and_crop_dataset
except ImportError:
    from config import (
        CROPPED_DATA_DIR,
        MODEL_PATH,
        SCALER_PCA_PATH,
        CLASS_DICT_PATH,
        INDIAN_CRICKETERS,
        PLAYER_DISPLAY_NAMES
    )
    from preprocessor import FeatureExtractor
    from fetch_dataset import download_and_crop_dataset


class ModelTrainer:
    """
    Automated Training and Model Selection for Cricket Face Recognition.
    """

    def __init__(self, cropped_dir: Path = CROPPED_DATA_DIR):
        self.cropped_dir = cropped_dir
        self.feature_extractor = FeatureExtractor()
        self.class_dict = {}
        self.inverse_class_dict = {}

    def load_dataset_features(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Loads all cropped face images from disk and extracts combined feature vectors.
        """
        player_dirs = sorted([d for d in self.cropped_dir.iterdir() if d.is_dir() and list(d.glob("*.jpg"))])
        
        # If dataset is incomplete or empty, fetch real photos
        if len(player_dirs) < 2:
            print("[*] Insufficient face dataset found. Running real photo fetcher...")
            download_and_crop_dataset()
            player_dirs = sorted([d for d in self.cropped_dir.iterdir() if d.is_dir() and list(d.glob("*.jpg"))])

        X_list = []
        y_list = []
        self.class_dict = {}
        current_id = 0

        for p_dir in player_dirs:
            player_slug = p_dir.name
            img_files = list(p_dir.glob("*.jpg")) + list(p_dir.glob("*.png"))
            
            if not img_files:
                continue

            self.class_dict[player_slug] = current_id
            print(f"[*] Processing player '{player_slug}' (Class ID: {current_id}) - {len(img_files)} images")

            for img_path in img_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                feat = self.feature_extractor.extract_combined_features(img)
                if feat is not None:
                    X_list.append(feat)
                    y_list.append(current_id)

            current_id += 1

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        self.inverse_class_dict = {v: k for k, v in self.class_dict.items()}

        # Save class dictionary to disk
        with open(str(CLASS_DICT_PATH), "w") as f:
            json.dump(self.class_dict, f, indent=2)

        print(f"[✓] Feature Extraction Complete: {X.shape[0]} total samples across {len(self.class_dict)} players (Feature Dim: {X.shape[1]})")
        return X, y, self.class_dict

    def train_and_tune(self, test_size: float = 0.20, random_state: int = 42) -> Dict[str, Any]:
        """
        Trains classifiers with GridSearchCV, calculates class centroids,
        and saves the complete model artifact.
        """
        X, y, class_dict = self.load_dataset_features()
        if len(X) == 0:
            raise ValueError("No training samples found in dataset!")

        if len(class_dict) < 2:
            raise ValueError(f"Need at least 2 distinct player classes to train, found {len(class_dict)}")

        # Calculate Normalized Class Centroids for Cosine Similarity Matching
        centroids = {}
        for c_id in range(len(class_dict)):
            class_samples = X[y == c_id]
            if len(class_samples) > 0:
                mean_vec = np.mean(class_samples, axis=0)
                mean_vec /= (np.linalg.norm(mean_vec) + 1e-6)
                centroids[int(c_id)] = mean_vec

        # Train / Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"[*] Split into {len(X_train)} training samples and {len(X_test)} validation samples.")

        # Candidate Model Grids
        model_params = {
            "SVM (RBF Kernel)": {
                "model": SVC(probability=True, random_state=random_state),
                "params": {
                    "classifier__C": [0.1, 1, 10, 50],
                    "classifier__gamma": ["scale", "auto", 0.01],
                    "classifier__kernel": ["rbf"]
                }
            },
            "SVM (Linear)": {
                "model": SVC(probability=True, random_state=random_state),
                "params": {
                    "classifier__C": [0.1, 1, 10],
                    "classifier__kernel": ["linear"]
                }
            },
            "Random Forest": {
                "model": RandomForestClassifier(random_state=random_state),
                "params": {
                    "classifier__n_estimators": [50, 100],
                    "classifier__max_depth": [None, 15]
                }
            },
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=1000, random_state=random_state),
                "params": {
                    "classifier__C": [0.1, 1.0, 10.0]
                }
            },
            "K-Nearest Neighbors": {
                "model": KNeighborsClassifier(),
                "params": {
                    "classifier__n_neighbors": [3, 5],
                    "classifier__weights": ["distance"]
                }
            }
        }

        min_class_samples = np.min(np.bincount(y_train))
        cv_folds = min(3, max(2, min_class_samples))
        n_components = min(35, len(X_train) - 1, X.shape[1])

        print(f"[*] Starting GridSearchCV with {cv_folds}-Fold Cross-Validation (PCA Components: {n_components})...")

        best_score = -1.0
        best_model_name = ""
        best_pipeline = None
        benchmark_results = {}

        for name, config in model_params.items():
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=random_state)),
                ("classifier", config["model"])
            ])

            grid = GridSearchCV(
                pipe,
                config["params"],
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
                scoring="accuracy",
                n_jobs=-1
            )
            grid.fit(X_train, y_train)

            test_preds = grid.predict(X_test)
            test_acc = accuracy_score(y_test, test_preds)

            benchmark_results[name] = {
                "cv_best_score": round(float(grid.best_score_ * 100), 2),
                "test_accuracy": round(float(test_acc * 100), 2),
                "best_params": grid.best_params_
            }

            print(f"  [+] {name:22} | CV Score: {grid.best_score_*100:6.2f}% | Test Acc: {test_acc*100:6.2f}%")

            if test_acc > best_score:
                best_score = test_acc
                best_model_name = name
                best_pipeline = grid.best_estimator_

        print(f"\n[🏆] Winning Model: {best_model_name} with {best_score*100:.2f}% Test Accuracy!")

        # Fit best pipeline on full dataset
        best_pipeline.fit(X, y)

        # Save package with model + class centroids for hybrid cosine inference
        model_payload = {
            "model": best_pipeline,
            "centroids": centroids
        }
        joblib.dump(model_payload, str(MODEL_PATH))
        print(f"[✓] Saved best trained model to: {MODEL_PATH}")

        return {
            "best_model_name": best_model_name,
            "best_accuracy": best_score,
            "benchmark_results": benchmark_results,
            "classes_count": len(class_dict),
            "total_samples": len(X)
        }


def main():
    trainer = ModelTrainer()
    results = trainer.train_and_tune()


if __name__ == "__main__":
    main()
