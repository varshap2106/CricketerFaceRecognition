"""
Dataset Collection and Preprocessing Pipeline.
Provides automated image downloaders, Kaggle dataset ingestors, and starter dataset generator.
"""

import os
import re
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import cv2

try:
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
except ImportError:
    from config import (
        INDIAN_CRICKETERS,
        PLAYER_DISPLAY_NAMES,
        RAW_DATA_DIR,
        CROPPED_DATA_DIR,
        STARTER_DATA_DIR,
        FACE_SIZE
    )
    from face_detector import FaceDetector
    from preprocessor import DataAugmentor


class DatasetManager:
    """
    Manages dataset acquisition, face cropping, organization, and validation.
    """

    def __init__(self):
        self.detector = FaceDetector()

    def download_player_images_web(
        self,
        player_name: str,
        limit: int = 15,
        output_dir: Optional[Path] = None
    ) -> int:
        """
        Downloads face images of a specific cricketer from web search.
        """
        slug = player_name.lower().replace(" ", "_")
        target_dir = (output_dir or RAW_DATA_DIR) / slug
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"[*] Searching and fetching images for: {player_name} (Target: {limit} images)...")
        
        # Build search queries
        query = f"{player_name} Indian cricket player face portrait"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }

        downloaded_count = 0
        try:
            # DuckDuckGo HTML image search query
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode("utf-8", errors="ignore")
                
            # Extract image URLs
            img_urls = re.findall(r'src="(https?://[^"]+\.(?:jpg|jpeg|png))"', html)
            if not img_urls:
                img_urls = re.findall(r'img_url=(https?%3A%2F%2F[^&]+)', html)
                img_urls = [urllib.parse.unquote(u) for u in img_urls]

            for idx, img_url in enumerate(img_urls[:limit * 2]):
                if downloaded_count >= limit:
                    break
                try:
                    save_path = target_dir / f"{slug}_{downloaded_count + 1:03d}.jpg"
                    img_req = urllib.request.Request(img_url, headers=headers)
                    with urllib.request.urlopen(img_req, timeout=5) as img_resp:
                        img_bytes = bytearray(img_resp.read())
                        img_arr = np.asarray(img_bytes, dtype=np.uint8)
                        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                        if img is not None and img.shape[0] > 100 and img.shape[1] > 100:
                            cv2.imwrite(str(save_path), img)
                            downloaded_count += 1
                except Exception:
                    continue

        except Exception as e:
            print(f"[-] Web search notice for {player_name}: {e}")

        print(f"[✓] Downloaded {downloaded_count} raw images for {player_name} in {target_dir}")
        return downloaded_count

    def crop_all_raw_dataset(
        self,
        raw_dir: Path = RAW_DATA_DIR,
        cropped_dir: Path = CROPPED_DATA_DIR,
        augment: bool = True
    ) -> Dict[str, int]:
        """
        Iterates over all raw folders, runs face detection, and saves aligned crops to cropped_dir.
        Optionally generates augmented copies for better model generalization.
        """
        stats = {}
        cropped_dir.mkdir(parents=True, exist_ok=True)

        if not raw_dir.exists():
            print(f"[-] Raw directory not found: {raw_dir}")
            return stats

        player_folders = [f for f in raw_dir.iterdir() if f.is_dir()]
        print(f"[*] Processing {len(player_folders)} player folders for face cropping...")

        for folder in player_folders:
            player_slug = folder.name
            target_player_dir = cropped_dir / player_slug
            target_player_dir.mkdir(parents=True, exist_ok=True)

            image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
            saved_count = 0

            for img_path in image_files:
                crops = self.detector.detect_and_crop_faces(img_path, require_eyes=False, target_size=FACE_SIZE)
                
                # If face detection didn't trigger, attempt center crop as fallback
                if len(crops) == 0:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        min_dim = min(h, w)
                        cy, cx = h // 2, w // 2
                        center_crop = img[cy - min_dim//2 : cy + min_dim//2, cx - min_dim//2 : cx + min_dim//2]
                        resized = cv2.resize(center_crop, FACE_SIZE)
                        crops = [(resized, (0, 0, w, h))]

                for crop, _ in crops:
                    if augment:
                        augmented_list = DataAugmentor.augment_image(crop)
                        for aug_idx, aug_img in enumerate(augmented_list):
                            saved_count += 1
                            out_filename = f"{player_slug}_crop_{saved_count:04d}.jpg"
                            cv2.imwrite(str(target_player_dir / out_filename), aug_img)
                    else:
                        saved_count += 1
                        out_filename = f"{player_slug}_crop_{saved_count:04d}.jpg"
                        cv2.imwrite(str(target_player_dir / out_filename), crop)

            stats[player_slug] = saved_count
            print(f"[✓] Cropped & saved {saved_count} face samples for '{player_slug}'")

        return stats

    def generate_starter_dataset(self, samples_per_player: int = 25) -> Dict[str, int]:
        """
        Creates a high-quality starter face dataset with distinctive facial signatures
        for all 10 Indian cricketers so the full pipeline can train and test instantly.
        """
        print("[*] Generating instant starter face dataset for Indian Cricket Team...")
        stats = {}
        CROPPED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Player-specific unique aesthetic traits & color hues for procedural visual signatures
        signatures = {
            "virat_kohli": {"base_hue": 12, "beard": True, "skin_tone": (145, 175, 215), "eye_y": 62, "jaw": 130},
            "rohit_sharma": {"base_hue": 14, "beard": True, "skin_tone": (140, 170, 210), "eye_y": 65, "jaw": 128},
            "ms_dhoni": {"base_hue": 15, "beard": True, "skin_tone": (135, 168, 205), "eye_y": 64, "jaw": 126},
            "sachin_tendulkar": {"base_hue": 16, "beard": False, "skin_tone": (150, 180, 220), "eye_y": 66, "jaw": 124},
            "jasprit_bumrah": {"base_hue": 11, "beard": True, "skin_tone": (125, 155, 195), "eye_y": 60, "jaw": 132},
            "hardik_pandya": {"base_hue": 10, "beard": True, "skin_tone": (120, 150, 190), "eye_y": 61, "jaw": 134},
            "ravindra_jadeja": {"base_hue": 13, "beard": True, "skin_tone": (130, 160, 200), "eye_y": 63, "jaw": 131},
            "shubman_gill": {"base_hue": 15, "beard": False, "skin_tone": (155, 185, 225), "eye_y": 64, "jaw": 125},
            "kl_rahul": {"base_hue": 13, "beard": True, "skin_tone": (138, 168, 208), "eye_y": 63, "jaw": 129},
            "rishabh_pant": {"base_hue": 14, "beard": True, "skin_tone": (142, 172, 212), "eye_y": 65, "jaw": 127}
        }

        for player_slug in INDIAN_CRICKETERS:
            target_dir = CROPPED_DATA_DIR / player_slug
            target_dir.mkdir(parents=True, exist_ok=True)
            sig = signatures.get(player_slug, signatures["virat_kohli"])
            
            for i in range(samples_per_player):
                # Synthesize 160x160 face image with realistic gradients and facial structure
                img = np.zeros((160, 160, 3), dtype=np.uint8)
                
                # Background gradient
                bg_color = (30 + (i * 3) % 20, 25 + (i * 2) % 15, 20 + i % 10)
                img[:] = bg_color

                # Face oval
                center = (80, 85 + (i % 3 - 1))
                axes = (48 + (i % 2), 62 + (i % 3))
                skin = tuple(int(c + np.random.randint(-8, 9)) for c in sig["skin_tone"])
                cv2.ellipse(img, center, axes, 0, 0, 360, skin, -1)

                # Hair / Cap
                hair_color = (20, 18, 15)
                cv2.ellipse(img, (80, 48), (48, 28), 0, 180, 360, hair_color, -1)
                
                # Eyes
                eye_y = sig["eye_y"] + (i % 3 - 1)
                cv2.ellipse(img, (60, eye_y), (8, 5), 0, 0, 360, (240, 240, 240), -1)
                cv2.circle(img, (60, eye_y), 3, (30, 20, 10), -1)
                cv2.ellipse(img, (100, eye_y), (8, 5), 0, 0, 360, (240, 240, 240), -1)
                cv2.circle(img, (100, eye_y), 3, (30, 20, 10), -1)

                # Eyebrows
                cv2.line(img, (50, eye_y - 8), (70, eye_y - 9), hair_color, 2)
                cv2.line(img, (90, eye_y - 9), (110, eye_y - 8), hair_color, 2)

                # Nose
                cv2.line(img, (80, eye_y), (80, eye_y + 20), (int(skin[0]*0.85), int(skin[1]*0.85), int(skin[2]*0.85)), 2)
                cv2.ellipse(img, (80, eye_y + 20), (7, 4), 0, 0, 180, (int(skin[0]*0.75), int(skin[1]*0.75), int(skin[2]*0.75)), 2)

                # Mouth
                mouth_y = eye_y + 36
                cv2.ellipse(img, (80, mouth_y), (14, 6), 0, 0, 180, (90, 95, 175), -1)

                # Beard / Stubble if player has one
                if sig["beard"]:
                    beard_color = (40, 35, 30)
                    cv2.ellipse(img, (80, sig["jaw"]), (38, 22), 0, 0, 180, beard_color, 4)

                # Add natural noise
                noise = np.random.normal(0, 3, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)

                # Smooth slightly
                img = cv2.GaussianBlur(img, (3, 3), 0.5)

                filename = f"{player_slug}_sample_{i+1:03d}.jpg"
                cv2.imwrite(str(target_dir / filename), img)

            stats[player_slug] = samples_per_player

        print(f"[✓] Successfully generated starter dataset for {len(stats)} players ({samples_per_player} samples each) in {CROPPED_DATA_DIR}")
        return stats
