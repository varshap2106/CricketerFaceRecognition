"""
Real-time Webcam Face Identification Stream with Cyber HUD and Player Card Overlays.
"""

import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from src.config import OUTPUTS_DIR, FACE_SIZE
    from src.predict import FacePredictor
except ImportError:
    from config import OUTPUTS_DIR, FACE_SIZE
    from predict import FacePredictor


class RealtimeFaceRecognition:
    """
    Real-time video feed processor for Indian Cricket Team Face Identification.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.predictor = FacePredictor()
        self.snapshot_dir = OUTPUTS_DIR / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def run(self, max_frames: Optional[int] = None):
        """
        Starts the real-time webcam face recognition loop.
        Press 'q' or ESC to exit.
        Press 's' to save a snapshot with annotations.
        """
        if not self.predictor.is_model_ready():
            print("[!] Model not found or trained. Please train the model first using 'python run_project.py train'")
            return

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[-] Could not access webcam at index {self.camera_index}. Please check device camera permissions.")
            return

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("===============================================================")
        print("  🏏 INDIAN CRICKET TEAM FACE IDENTIFICATION - LIVE WEBCAM HUD")
        print("  Controls:")
        print("    [Q] / [ESC] : Exit Application")
        print("    [S]         : Capture and Save Annotated Snapshot")
        print("===============================================================")

        prev_time = time.time()
        fps = 0.0
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[-] Failed to grab frame from webcam.")
                break

            frame_count += 1
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
            prev_time = curr_time

            # 1. Run Prediction Pipeline
            result = self.predictor.predict_image(frame)
            display_frame = result["annotated_image"]

            # 2. Draw Top Control Panel HUD
            h, w = display_frame.shape[:2]
            
            # Glass panel background at top
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (15, 15), (380, 95), (15, 20, 28), -1)
            cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)
            cv2.rectangle(display_frame, (15, 15), (380, 95), (0, 200, 255), 1)

            # HUD Text
            cv2.putText(display_frame, "INDIAN CRICKETER RECOGNITION", (25, 40), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 210, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"FPS: {fps:.1f} | Faces: {result['faces_detected']}", (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 220, 240), 1, cv2.LINE_AA)
            cv2.putText(display_frame, "[S] Snapshot  |  [Q] Exit", (25, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 160, 180), 1, cv2.LINE_AA)

            # Show active window
            cv2.imshow("Indian Cricket Team Face Recognition - Live Feed", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:  # 'q' or ESC
                break
            elif key in [ord('s'), ord('S')]:
                snapshot_path = self.snapshot_dir / f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(str(snapshot_path), display_frame)
                print(f"[✓] Saved snapshot to: {snapshot_path}")

            if max_frames and frame_count >= max_frames:
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[*] Live stream stopped.")


def main():
    app = RealtimeFaceRecognition()
    app.run()


if __name__ == "__main__":
    main()
