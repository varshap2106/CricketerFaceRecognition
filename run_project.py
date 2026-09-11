"""
Unified CLI Entry Point for Indian Cricket Team Face Identification System.
Provides commands to download data, preprocess faces, train models, evaluate, run web apps, and live webcam recognition.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to Python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    INDIAN_CRICKETERS,
    PLAYER_DISPLAY_NAMES,
    CROPPED_DATA_DIR,
    RAW_DATA_DIR,
    MODEL_PATH,
    OUTPUTS_DIR
)
from src.dataset_downloader import DatasetManager
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.predict import FacePredictor
from src.realtime import RealtimeFaceRecognition
from src.setup_real_model import build_real_cricket_model


def cmd_download(args):
    """Download images for one or all players."""
    dm = DatasetManager()
    if args.player:
        player_name = args.player
        dm.download_player_images_web(player_name, limit=args.limit)
    else:
        print(f"[*] Downloading images for all {len(INDIAN_CRICKETERS)} Indian cricket players...")
        for slug in INDIAN_CRICKETERS:
            name = PLAYER_DISPLAY_NAMES.get(slug, slug)
            dm.download_player_images_web(name, limit=args.limit)
    print("[✓] Download process finished.")


def cmd_crop(args):
    """Detect faces and crop all raw images."""
    dm = DatasetManager()
    stats = dm.crop_all_raw_dataset(augment=not args.no_augment)
    print("\n--- Cropping Summary ---")
    for player, count in stats.items():
        print(f"  {player:<20}: {count} face crops saved")


def cmd_train(args):
    """Train ML models using GridSearchCV and save best classifier."""
    print("===============================================================")
    print("  🏏 TRAINING INDIAN CRICKETER FACE RECOGNITION MODELS")
    print("===============================================================")
    trainer = ModelTrainer()
    results = trainer.train_and_tune(test_size=args.test_split)
    print(f"\n[🏆] Winning Model: {results['best_model_name']} with {results['best_accuracy']*100:.2f}% Test Accuracy!")


def cmd_evaluate(args):
    """Evaluate trained model and generate confusion matrix heatmap."""
    print("===============================================================")
    print("  📊 EVALUATING MODEL & GENERATING PERFORMANCE METRICS")
    print("===============================================================")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model()


def cmd_predict(args):
    """Run prediction on a single image file."""
    if not args.image:
        print("[-] Error: Please specify --image <path/to/image.jpg>")
        return

    predictor = FacePredictor()
    if not predictor.is_model_ready():
        print("[!] Model not found. Running training first...")
        build_real_cricket_model()
        predictor = FacePredictor()

    print(f"[*] Processing image: {args.image}...")
    results = predictor.predict_image(args.image)

    print(f"\n[✓] Detected {results['faces_detected']} face(s):")
    for idx, det in enumerate(results.get("detections", [])):
        p_name = det.get("player_name", "Unknown")
        conf = det.get("confidence", 0.0)
        profile = det.get("profile", {})
        print(f"  [Face {idx+1}] => {p_name} (Confidence: {conf*100:.1f}%)")
        if profile:
            print(f"            Role: {profile.get('role', 'N/A')}")
            print(f"            Jersey: No. {profile.get('jersey_no', 'N/A')}")
            print(f"            Runs: {profile.get('runs', 0):,} | 100s: {profile.get('centuries', 0)}")
            print(f"            Accolade: {profile.get('key_achievements', 'N/A')}")

    if args.output:
        import cv2
        cv2.imwrite(args.output, results["annotated_image"])
        print(f"[✓] Saved annotated image to: {args.output}")


def cmd_realtime(args):
    """Start real-time webcam face recognition."""
    app = RealtimeFaceRecognition(camera_index=args.camera)
    app.run()


def cmd_web(args):
    """Launch the modern Flask web application."""
    from web_app.app import app
    port = args.port
    print(f"[*] Starting Indian Cricket Face Recognition Web Dashboard on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=args.debug)


def cmd_streamlit(args):
    """Launch the Streamlit interactive studio."""
    print("[*] Launching Streamlit Interactive Studio...")
    os.system("streamlit run streamlit_app.py")


def cmd_init_demo(args):
    """One-command full setup: downloads real cricketer photos, trains ML model on real faces, and produces confusion matrix."""
    print("===============================================================")
    print("  🚀 INITIALIZING FULL REAL INDIAN CRICKETER FACE AI SYSTEM")
    print("===============================================================")
    from src.fetch_dataset import download_and_crop_dataset
    download_and_crop_dataset()

    trainer = ModelTrainer()
    benchmark = trainer.train_and_tune(test_size=0.20)

    evaluator = ModelEvaluator()
    evaluator.evaluate_model()

    print("\n[🎉] Complete real-world system initialized successfully!")
    print("Run web app:       python run_project.py web")
    print("Run streamlit:     python run_project.py streamlit")
    print("Run live webcam:   python run_project.py realtime")


def main():
    parser = argparse.ArgumentParser(
        description="🏏 Indian Cricket Team Member Face Identification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_project.py init-demo
  python run_project.py web --port 5000
  python run_project.py streamlit
  python run_project.py realtime
  python run_project.py predict --image sample.jpg
  python run_project.py train
  python run_project.py evaluate
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # init-demo
    subparsers.add_parser("init-demo", help="One-command full setup (downloads real images, trains model, produces metrics)")

    # download
    p_dl = subparsers.add_parser("download", help="Download player images from web search")
    p_dl.add_argument("--player", type=str, help="Name of the player (e.g., 'Virat Kohli')")
    p_dl.add_argument("--limit", type=int, default=15, help="Number of images to download per player")

    # crop
    p_crop = subparsers.add_parser("crop", help="Detect faces and crop raw images")
    p_crop.add_argument("--no-augment", action="store_true", help="Disable data augmentation")

    # train
    p_train = subparsers.add_parser("train", help="Train models with GridSearchCV")
    p_train.add_argument("--test-split", type=float, default=0.20, help="Validation test split ratio")

    # evaluate
    subparsers.add_parser("evaluate", help="Compute metrics and plot confusion matrix heatmap")

    # predict
    p_pred = subparsers.add_parser("predict", help="Predict player for an image")
    p_pred.add_argument("--image", type=str, required=True, help="Path to input image file")
    p_pred.add_argument("--output", type=str, help="Path to save annotated output image")

    # realtime
    p_rt = subparsers.add_parser("realtime", help="Run live webcam face recognition HUD")
    p_rt.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")

    # web
    p_web = subparsers.add_parser("web", help="Launch modern Flask web dashboard")
    p_web.add_argument("--port", type=int, default=5000, help="Port to run web server on")
    p_web.add_argument("--debug", action="store_true", help="Run in debug mode")

    # streamlit
    subparsers.add_parser("streamlit", help="Launch Streamlit interactive ML studio")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "init-demo": cmd_init_demo,
        "download": cmd_download,
        "crop": cmd_crop,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
        "realtime": cmd_realtime,
        "web": cmd_web,
        "streamlit": cmd_streamlit
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
