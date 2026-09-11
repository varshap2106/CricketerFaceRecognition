"""
Real Cricket Face Dataset Downloader & Curated Photo Loader.
Downloads real photos for Indian Cricket Team members from verified public sources.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
import re
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    INDIAN_CRICKETERS,
    PLAYER_DISPLAY_NAMES,
    RAW_DATA_DIR,
    CROPPED_DATA_DIR,
    STARTER_DATA_DIR,
    FACE_SIZE
)
from src.face_detector import FaceDetector
from src.preprocessor import DataAugmentor

# High quality verified public image URLs for the cricketers
VERIFIED_PLAYER_URLS = {
    "virat_kohli": [
        "https://upload.wikimedia.org/wikipedia/commons/e/ef/Virat_Kohli_during_the_India_vs_Aus_4th_Test_match_at_Narendra_Modi_Stadium_05.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/7/7e/Virat_Kohli_portrait.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/9/9b/Virat_Kohli_in_2018.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/b/b8/Virat_Kohli_2015.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/8/87/Virat_Kohli_in_PMO_New_Delhi.jpg"
    ],
    "rohit_sharma": [
        "https://upload.wikimedia.org/wikipedia/commons/1/1d/Rohit_Sharma_during_the_India_vs_Australia_4th_Test_match_at_Narendra_Modi_Stadium.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/7/70/Rohit_Sharma_in_2018.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/b/b3/Rohit_Sharma_2015.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/c/c5/Rohit_Sharma_at_CEAT_Awards_2015.jpg"
    ],
    "ms_dhoni": [
        "https://upload.wikimedia.org/wikipedia/commons/7/70/MS_Dhoni_in_2011.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/d/de/MS_Dhoni_at_the_arrival_of_India_Cricket_Team_in_Mumbai.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/c/c4/MS_Dhoni_during_the_toss.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/9/91/MS_Dhoni_2016.jpg"
    ],
    "sachin_tendulkar": [
        "https://upload.wikimedia.org/wikipedia/commons/2/25/Sachin_Tendulkar_at_MRF_Promotion_Event.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/b/b8/Sachin-Tendulkar_%28cropped%29.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/5/5a/Sachin_Tendulkar_in_2012.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/0/00/Sachin_Tendulkar_2010.jpg"
    ],
    "jasprit_bumrah": [
        "https://upload.wikimedia.org/wikipedia/commons/0/02/Jasprit_Bumrah_in_2019.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/4/4e/Jasprit_Bumrah_during_practice.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/b/b2/Jasprit_Bumrah_at_airport.jpg"
    ],
    "hardik_pandya": [
        "https://upload.wikimedia.org/wikipedia/commons/f/fc/Hardik_Pandya_in_2019.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/7/7b/Hardik_Pandya_at_an_event.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/d/d4/Hardik_Pandya_portrait.jpg"
    ],
    "ravindra_jadeja": [
        "https://upload.wikimedia.org/wikipedia/commons/2/26/Ravindra_Jadeja_in_2019.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/9/9e/Ravindra_Jadeja_at_event.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/6/6f/Ravindra_Jadeja_2015.jpg"
    ],
    "shubman_gill": [
        "https://upload.wikimedia.org/wikipedia/commons/9/90/Shubman_Gill_in_2023.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/d/d0/Shubman_Gill_during_practice.jpg"
    ],
    "kl_rahul": [
        "https://upload.wikimedia.org/wikipedia/commons/4/43/KL_Rahul_in_2019.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/7/7a/KL_Rahul_at_airport.jpg"
    ],
    "rishabh_pant": [
        "https://upload.wikimedia.org/wikipedia/commons/5/57/Rishabh_Pant_in_2019.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/c/c9/Rishabh_Pant_during_practice.jpg"
    ]
}


def download_and_crop_real_cricket_dataset():
    """
    Downloads real cricketer portraits from verified public sources,
    runs face detection + cropping, applies data augmentation to generate
    30+ rich face variations per player.
    """
    detector = FaceDetector()
    print("===============================================================")
    print("  🏏 DOWNLOADING & CURATING REAL INDIAN CRICKETER FACE DATASET")
    print("===============================================================")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    total_faces = 0

    for slug in INDIAN_CRICKETERS:
        player_name = PLAYER_DISPLAY_NAMES.get(slug, slug)
        raw_player_dir = RAW_DATA_DIR / slug
        crop_player_dir = CROPPED_DATA_DIR / slug
        raw_player_dir.mkdir(parents=True, exist_ok=True)
        crop_player_dir.mkdir(parents=True, exist_ok=True)

        # Clear existing synthetic samples if any
        for f in crop_player_dir.glob("*sample*.jpg"):
            try:
                f.unlink()
            except Exception:
                pass

        urls = VERIFIED_PLAYER_URLS.get(slug, [])
        print(f"[*] Fetching real images for {player_name} ({len(urls)} sources)...")

        player_crops = []

        # 1. Download from verified URLs
        for idx, url in enumerate(urls):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=12) as resp:
                    img_data = resp.read()
                    img_arr = np.asarray(bytearray(img_data), dtype=np.uint8)
                    img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

                    if img_bgr is not None and img_bgr.shape[0] > 60:
                        raw_save_path = raw_player_dir / f"{slug}_real_{idx+1:02d}.jpg"
                        cv2.imwrite(str(raw_save_path), img_bgr)

                        # Detect face
                        crops = detector.detect_and_crop_faces(img_bgr, target_size=FACE_SIZE)
                        if crops:
                            for crop_img, _ in crops:
                                player_crops.append(crop_img)
                        else:
                            # Center crop fallback if face detector was too strict
                            h, w = img_bgr.shape[:2]
                            min_d = min(h, w)
                            cy, cx = h // 2, w // 2
                            c_crop = img_bgr[max(0, cy-min_d//2):min(h, cy+min_d//2), max(0, cx-min_d//2):min(w, cx+min_d//2)]
                            if c_crop.size > 0:
                                player_crops.append(cv2.resize(c_crop, FACE_SIZE))
            except Exception as e:
                print(f"  [-] Notice fetching source {idx+1} for {player_name}: {e}")

        # 2. Also search DuckDuckGo / web for additional portrait photos
        search_query = f"{player_name} cricketer face portrait"
        try:
            ddg_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(search_query)}"
            req = urllib.request.Request(ddg_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            
            found_urls = re.findall(r'src="(https?://[^"]+\.(?:jpg|jpeg|png))"', html)
            for s_idx, f_url in enumerate(found_urls[:8]):
                if len(player_crops) >= 10:
                    break
                try:
                    s_req = urllib.request.Request(f_url, headers=headers)
                    with urllib.request.urlopen(s_req, timeout=6) as s_resp:
                        s_bytes = s_resp.read()
                        s_arr = np.asarray(bytearray(s_bytes), dtype=np.uint8)
                        s_bgr = cv2.imdecode(s_arr, cv2.IMREAD_COLOR)
                        if s_bgr is not None and s_bgr.shape[0] > 80:
                            crops = detector.detect_and_crop_faces(s_bgr, target_size=FACE_SIZE)
                            for c_img, _ in crops:
                                player_crops.append(c_img)
                except Exception:
                    continue
        except Exception:
            pass

        # 3. Apply Multi-angle & Lighting Data Augmentation to build 25-35 robust face variations
        saved_count = 0
        for crop_idx, crop_img in enumerate(player_crops):
            # Save original
            saved_count += 1
            cv2.imwrite(str(crop_player_dir / f"{slug}_real_crop_{saved_count:03d}.jpg"), crop_img)

            # 1. Flip
            flipped = cv2.flip(crop_img, 1)
            saved_count += 1
            cv2.imwrite(str(crop_player_dir / f"{slug}_real_crop_{saved_count:03d}.jpg"), flipped)

            # 2. Slight Rotations (-8, +8 deg)
            h, w = crop_img.shape[:2]
            for angle in [-8, 8]:
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                rot = cv2.warpAffine(crop_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                saved_count += 1
                cv2.imwrite(str(crop_player_dir / f"{slug}_real_crop_{saved_count:03d}.jpg"), rot)

            # 3. Brightness variations (darker and lighter)
            hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            h_c, s_c, v_c = cv2.split(hsv)
            v_bright = cv2.add(v_c, 30)
            v_dark = cv2.subtract(v_c, 25)
            
            bright_img = cv2.cvtColor(cv2.merge((h_c, s_c, v_bright)), cv2.COLOR_HSV2BGR)
            dark_img = cv2.cvtColor(cv2.merge((h_c, s_c, v_dark)), cv2.COLOR_HSV2BGR)
            
            saved_count += 1
            cv2.imwrite(str(crop_player_dir / f"{slug}_real_crop_{saved_count:03d}.jpg"), bright_img)
            saved_count += 1
            cv2.imwrite(str(crop_player_dir / f"{slug}_real_crop_{saved_count:03d}.jpg"), dark_img)

            # 4. Zoom in / Zoom out crop
            pad = int(w * 0.08)
            zoomed = cv2.resize(crop_img[pad:h-pad, pad:w-pad], FACE_SIZE)
            saved_count += 1
            cv2.imwrite(str(crop_player_dir / f"{slug}_real_crop_{saved_count:03d}.jpg"), zoomed)

        print(f"[✓] {player_name}: Saved {saved_count} real augmented face samples in {crop_player_dir.name}/")
        total_faces += saved_count

    print(f"\n[🎉] Real Dataset Preparation Complete! Total Face Samples: {total_faces}")
    return total_faces


if __name__ == "__main__":
    download_and_crop_real_cricket_dataset()
