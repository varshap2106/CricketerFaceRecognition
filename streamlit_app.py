"""
Streamlit Web Application for Indian Cricket Team Face Identification.
Provides an interactive Machine Learning studio for face recognition,
dataset exploration, model retraining, and evaluation visualizations.
"""

import os
import io
import json
import time
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup Path
import sys
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    INDIAN_CRICKETERS,
    PLAYER_DISPLAY_NAMES,
    DEFAULT_PLAYER_PROFILES,
    MODEL_PATH,
    CROPPED_DATA_DIR,
    OUTPUTS_DIR,
    FACE_SIZE
)
from src.face_detector import FaceDetector
from src.preprocessor import FeatureExtractor
from src.predict import FacePredictor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.dataset_downloader import DatasetManager

# Page Configuration
st.set_page_config(
    page_title="Indian Cricket Team Face Recognition",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #00d2d3;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #a4b0be;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(0, 210, 211, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .player-highlight {
        background: linear-gradient(135deg, #0b132b, #1c2541);
        border-left: 4px solid #00d2d3;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    return FacePredictor()


@st.cache_resource
def get_detector():
    return FaceDetector()


def main():
    st.sidebar.image("https://img.icons8.com/color/96/cricket.png", width=70)
    st.sidebar.title("🏏 CricVision Studio")
    st.sidebar.markdown("**Team India Face Identification**")
    
    app_mode = st.sidebar.radio(
        "Navigation",
        [
            "🎯 Face Identification",
            "📷 Live Webcam Scanner",
            "🔍 Dataset Explorer",
            "⚙️ Train & Tune Model",
            "📊 Evaluation Metrics",
            "👥 Indian Cricket Roster"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tech Stack**: OpenCV Haar Cascade, 2D Wavelet Decomposition, Scikit-Learn SVM/PCA, Streamlit.")

    predictor = get_predictor()
    detector = get_detector()

    # =========================================================
    # TAB 1: FACE IDENTIFICATION
    # =========================================================
    if app_mode == "🎯 Face Identification":
        st.markdown("<h1 class='main-title'>Indian Cricket Team Member Identification</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>Upload an image of an Indian cricketer or team photo for instant face detection and player recognition.</p>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.subheader("📤 Input Image")
            input_mode = st.radio("Choose Input Mode:", ["Upload File", "Select Preset Demo Sample"], horizontal=True)
            
            input_image_bgr = None

            if input_mode == "Upload File":
                uploaded_file = st.file_uploader("Upload Cricket Image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg", "webp"])
                if uploaded_file is not None:
                    image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    input_image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                    st.image(cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

            else:
                sample_player = st.selectbox(
                    "Select Indian Cricket Player Demo:",
                    list(PLAYER_DISPLAY_NAMES.values())
                )
                # Find matching slug
                slug = [k for k, v in PLAYER_DISPLAY_NAMES.items() if v == sample_player][0]
                player_folder = CROPPED_DATA_DIR / slug
                
                if player_folder.exists() and list(player_folder.glob("*.jpg")):
                    sample_file = list(player_folder.glob("*.jpg"))[0]
                    input_image_bgr = cv2.imread(str(sample_file))
                    st.image(cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB), caption=f"Demo Sample: {sample_player}", use_column_width=True)
                else:
                    st.warning("No sample files found yet. Go to '⚙️ Train & Tune Model' tab to generate dataset.")

            conf_thresh = st.slider("Confidence Threshold", min_value=0.20, max_value=0.95, value=0.35, step=0.05)

        with col2:
            st.subheader("🎯 Identification Output")
            if input_image_bgr is not None:
                if st.button("🚀 Identify Cricketer", type="primary", use_container_width=True):
                    with st.spinner("Processing facial landmarks and extracting Wavelet embeddings..."):
                        results = predictor.predict_image(input_image_bgr, confidence_threshold=conf_thresh)

                    st.success(f"Detected {results['faces_detected']} face(s) in image!")
                    st.image(
                        cv2.cvtColor(results['annotated_image'], cv2.COLOR_BGR2RGB),
                        caption="Annotated Recognition Result",
                        use_column_width=True
                    )

                    for idx, det in enumerate(results.get("detections", [])):
                        profile = det.get("profile", {})
                        p_name = det.get("player_name", "Unknown")
                        conf = det.get("confidence", 0.0)

                        st.markdown(f"""
                        <div class="player-highlight">
                            <h3>👑 Recognized: <span style="color:#00d2d3;">{p_name}</span> (Match: {int(conf*100)}%)</h3>
                            <p><strong>Role:</strong> {profile.get('role', 'N/A')} | <strong>Jersey:</strong> No. {profile.get('jersey_no', 'N/A')}</p>
                            <p><strong>Stats:</strong> Runs: {profile.get('runs', 0):,} | 100s: {profile.get('centuries', 0)} | Wickets: {profile.get('wickets', 0)}</p>
                            <p><em>"{profile.get('bio', '')}"</em></p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Top Predictions Bars
                        if det.get("top_predictions"):
                            st.write("##### Probability Confidence Distribution")
                            for top_p in det["top_predictions"]:
                                st.progress(min(1.0, float(top_p["confidence"])), text=f"{top_p['name']} - {int(top_p['confidence']*100)}%")
            else:
                st.info("👈 Please select or upload an image on the left panel.")

    # =========================================================
    # TAB 2: LIVE WEBCAM SCANNER
    # =========================================================
    elif app_mode == "📷 Live Webcam Scanner":
        st.markdown("<h1 class='main-title'>Live Camera Face Scanner</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>Take a snapshot using your webcam to run real-time face identification.</p>", unsafe_allow_html=True)

        camera_photo = st.camera_input("Capture Face from Webcam")
        if camera_photo is not None:
            bytes_data = camera_photo.getvalue()
            img_arr = np.frombuffer(bytes_data, np.uint8)
            img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            with st.spinner("Analyzing captured frame..."):
                results = predictor.predict_image(img_bgr)

            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(results['annotated_image'], cv2.COLOR_BGR2RGB), caption="Identified Snapshot", use_column_width=True)
            with col2:
                st.subheader("Player Identification Card")
                if results["detections"]:
                    top_det = results["detections"][0]
                    profile = top_det.get("profile", {})
                    st.metric("Identified Player", top_det["player_name"], f"{int(top_det['confidence']*100)}% Match")
                    st.write(f"**Role:** {profile.get('role', 'N/A')}")
                    st.write(f"**Batting Style:** {profile.get('batting_style', 'N/A')}")
                    st.write(f"**ICC Rank:** {profile.get('icc_rank', 'N/A')}")
                    st.write(f"**Key Accolade:** {profile.get('key_achievements', 'N/A')}")
                else:
                    st.warning("No face detected in snapshot. Ensure your face is well-lit and facing forward.")

    # =========================================================
    # TAB 3: DATASET EXPLORER
    # =========================================================
    elif app_mode == "🔍 Dataset Explorer":
        st.markdown("<h1 class='main-title'>Dataset Explorer & Face Preprocessor</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>Inspect the cropped and aligned 160x160 face datasets for all Indian cricketers.</p>", unsafe_allow_html=True)

        selected_player = st.selectbox("Select Player Dataset:", list(PLAYER_DISPLAY_NAMES.values()))
        slug = [k for k, v in PLAYER_DISPLAY_NAMES.items() if v == selected_player][0]
        player_folder = CROPPED_DATA_DIR / slug

        if player_folder.exists() and list(player_folder.glob("*.jpg")):
            images = list(player_folder.glob("*.jpg"))
            st.success(f"Found {len(images)} face samples for {selected_player}")

            cols = st.columns(5)
            for idx, img_path in enumerate(images[:15]):
                img = cv2.imread(str(img_path))
                with cols[idx % 5]:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Sample {idx+1}", use_column_width=True)
        else:
            st.warning(f"No cropped samples found for {selected_player}. Please click 'Generate Dataset' in the Training tab.")

    # =========================================================
    # TAB 4: TRAIN & TUNE MODEL
    # =========================================================
    elif app_mode == "⚙️ Train & Tune Model":
        st.markdown("<h1 class='main-title'>Model Training & Hyperparameter Tuning</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>Extract 2D Wavelet Transform features, apply PCA reduction, and run GridSearchCV across ML classifiers.</p>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training Parameters")
            test_split = st.slider("Validation Test Split Ratio", 0.10, 0.40, 0.20, 0.05)
            samples_gen = st.number_input("Starter Dataset Samples Per Player", min_value=10, max_value=50, value=25)

            if st.button("🚀 Initialize Dataset & Train Model", type="primary", use_container_width=True):
                with st.spinner("Training ML Models (SVM, Random Forest, Logistic Regression, KNN) with Cross-Validation..."):
                    dm = DatasetManager()
                    dm.generate_starter_dataset(samples_per_player=int(samples_gen))

                    trainer = ModelTrainer()
                    benchmark = trainer.train_and_tune(test_size=test_split)

                    evaluator = ModelEvaluator()
                    metrics = evaluator.evaluate_model()

                st.success(f"🏆 Best Model: {benchmark['best_model_name']} with {benchmark['best_accuracy']*100:.2f}% Accuracy!")

        with col2:
            st.subheader("Model Benchmark Results")
            rep_path = OUTPUTS_DIR / "evaluation_report.json"
            if rep_path.exists():
                with open(str(rep_path), "r") as f:
                    rep_data = json.load(f)
                st.metric("Overall System Accuracy", f"{rep_data['overall_accuracy']*100:.2f}%")
                st.metric("Indexed Indian Cricketers", rep_data["num_classes"])
                st.metric("Total Training Samples", rep_data["total_samples"])

    # =========================================================
    # TAB 5: EVALUATION METRICS
    # =========================================================
    elif app_mode == "📊 Evaluation Metrics":
        st.markdown("<h1 class='main-title'>Model Performance & Evaluation Charts</h1>", unsafe_allow_html=True)
        cm_path = OUTPUTS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Normalized Confusion Matrix (10 Indian Cricketers)", use_column_width=True)
        else:
            st.info("Run Model Training in the '⚙️ Train & Tune Model' tab to generate the confusion matrix.")

    # =========================================================
    # TAB 6: ROSTER
    # =========================================================
    elif app_mode == "👥 Indian Cricket Roster":
        st.markdown("<h1 class='main-title'>Team India Indexed Squad</h1>", unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, (slug, prof) in enumerate(DEFAULT_PLAYER_PROFILES.items()):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="player-highlight">
                    <h4>No. {prof['jersey_no']} — {prof['name']}</h4>
                    <p><strong>Nickname:</strong> {prof['nickname']}</p>
                    <p><strong>Role:</strong> {prof['role']}</p>
                    <p><strong>Runs:</strong> {prof['runs']:,} | <strong>100s:</strong> {prof['centuries']} | <strong>Wkts:</strong> {prof['wickets']}</p>
                    <p><strong>ICC Rank:</strong> {prof['icc_rank']}</p>
                    <p style="font-size:0.85rem; color:#cbd5e1;">{prof['bio']}</p>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
