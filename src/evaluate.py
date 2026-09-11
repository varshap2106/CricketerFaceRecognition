"""
Model Evaluation and Metrics Visualization Module.
Calculates Confusion Matrix, Classification Report, Precision/Recall/F1, and generates visual plots.
"""

import os
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Dict, Any

try:
    from src.config import (
        MODEL_PATH,
        CLASS_DICT_PATH,
        OUTPUTS_DIR,
        PLAYER_DISPLAY_NAMES
    )
    from src.train import ModelTrainer
except ImportError:
    from config import (
        MODEL_PATH,
        CLASS_DICT_PATH,
        OUTPUTS_DIR,
        PLAYER_DISPLAY_NAMES
    )
    from train import ModelTrainer


class ModelEvaluator:
    """
    Comprehensive evaluation of trained Cricket Face Identification models.
    """

    def __init__(self):
        self.trainer = ModelTrainer()
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    def evaluate_model(self) -> Dict[str, Any]:
        """
        Runs evaluation, computes confusion matrix, and generates plot artifacts in outputs/.
        """
        print("[*] Loading dataset features for model evaluation...")
        X, y, class_dict = self.trainer.load_dataset_features()

        if not MODEL_PATH.exists():
            print("[!] Model file not found. Running training first...")
            self.trainer.train_and_tune()

        loaded = joblib.load(str(MODEL_PATH))
        if isinstance(loaded, dict) and "model" in loaded:
            model = loaded["model"]
        else:
            model = loaded

        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)

        # Class Names
        inv_dict = {v: k for k, v in class_dict.items()}
        target_names = [PLAYER_DISPLAY_NAMES.get(inv_dict[i], inv_dict[i].replace("_", " ").title()) for i in range(len(class_dict))]

        # Classification Report
        report_dict = classification_report(y, y_pred, target_names=target_names, output_dict=True)
        report_text = classification_report(y, y_pred, target_names=target_names)

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 1. Plot Confusion Matrix
        plt.figure(figsize=(10, 8), dpi=150)
        plt.style.use("dark_background")
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            cbar=True,
            linewidths=0.5,
            linecolor="#1e272e"
        )
        plt.title(f"Confusion Matrix (Normalized) - Accuracy: {accuracy*100:.2f}%", fontsize=14, color="#00d2d3", pad=15)
        plt.xlabel("Predicted Player", fontsize=11, color="#ffffff")
        plt.ylabel("Actual Player", fontsize=11, color="#ffffff")
        plt.xticks(rotation=45, ha="right", color="#dfe4ea")
        plt.yticks(rotation=0, color="#dfe4ea")
        plt.tight_layout()
        
        cm_plot_path = OUTPUTS_DIR / "confusion_matrix.png"
        plt.savefig(str(cm_plot_path), bbox_inches="tight")
        plt.close()

        # 2. Save Metrics JSON
        metrics_data = {
            "overall_accuracy": round(float(accuracy), 4),
            "total_samples": len(X),
            "num_classes": len(class_dict),
            "classification_report": report_dict
        }
        report_json_path = OUTPUTS_DIR / "evaluation_report.json"
        with open(str(report_json_path), "w") as f:
            json.dump(metrics_data, f, indent=2)

        print("\n================ CLASSIFICATION REPORT ================")
        print(report_text)
        print(f"[✓] Overall Model Accuracy: {accuracy*100:.2f}%")
        print(f"[✓] Saved Confusion Matrix plot to: {cm_plot_path}")
        print(f"[✓] Saved Evaluation Report to: {report_json_path}")

        return metrics_data


def main():
    evaluator = ModelEvaluator()
    evaluator.evaluate_model()


if __name__ == "__main__":
    main()
