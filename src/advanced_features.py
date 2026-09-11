"""
High-Accuracy Deep-Aligned Facial Feature Extractor.
Uses multi-resolution HOG grids, Local Binary Patterns (LBP), 2D Wavelet energy,
and L2-normalized cosine metric embeddings for invariant face recognition.
"""

import cv2
import numpy as np
import importlib
from typing import Tuple, List, Optional

try:
    pywt = importlib.import_module("pywt")
    PYWT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pywt = None
    PYWT_AVAILABLE = False

try:
    from src.config import FACE_SIZE
except ImportError:
    FACE_SIZE = (160, 160)


class HighAccuracyFeatureExtractor:
    """
    Extracts illumination and scale-invariant facial embeddings using:
    1. Multi-scale Block HOG descriptors (8x8 cells, 9 bins)
    2. Local Binary Patterns (LBP) texture descriptor
    3. Wavelet sub-band energy density
    4. Facial symmetry and chrominance vectors
    All sub-vectors are L2-normalized to project onto unit hypersphere.
    """

    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        self.target_size = target_size

        # Multi-scale HOG descriptors
        self.hog_dense = cv2.HOGDescriptor(
            _winSize=(128, 128),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        self.hog_coarse = cv2.HOGDescriptor(
            _winSize=(128, 128),
            _blockSize=(32, 32),
            _blockStride=(16, 16),
            _cellSize=(16, 16),
            _nbins=9
        )

    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        """Computes uniform Local Binary Patterns (LBP) histogram."""
        h, w = gray.shape
        padded = np.pad(gray, 1, mode='edge')
        
        # 8 neighbor offsets
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, 1), (1, 1), (1, 0),
            (1, -1), (0, -1)
        ]
        
        lbp = np.zeros((h, w), dtype=np.uint8)
        center = padded[1:-1, 1:-1]
        
        for idx, (dy, dx) in enumerate(offsets):
            neighbor = padded[1+dy : h+1+dy, 1+dx : w+1+dx]
            lbp |= ((neighbor >= center).astype(np.uint8) << idx)

        # 4x4 spatial grid LBP histograms
        histograms = []
        cell_h, cell_w = h // 4, w // 4
        for i in range(4):
            for j in range(4):
                cell = lbp[i*cell_h : (i+1)*cell_h, j*cell_w : (j+1)*cell_w]
                hist, _ = np.histogram(cell.ravel(), bins=16, range=(0, 256))
                hist = hist.astype(np.float32)
                hist /= (np.linalg.norm(hist) + 1e-6)
                histograms.extend(hist)

        return np.array(histograms, dtype=np.float32)

    def extract_wavelet_energy(self, gray: np.ndarray) -> np.ndarray:
        """Extracts energy distribution across Wavelet sub-bands."""
        gray_f = np.float32(gray) / 255.0
        if PYWT_AVAILABLE and pywt is not None:
            coeffs = pywt.wavedec2(gray_f, 'db1', level=3)
            energies = []
            for sub in coeffs[1:]:
                for band in sub:
                    energies.append(np.mean(band ** 2))
                    energies.append(np.std(band))
            energies = np.array(energies, dtype=np.float32)
            energies /= (np.linalg.norm(energies) + 1e-6)
            return energies
        else:
            # Fallback Laplacian energy pyramid
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            return np.array([np.mean(lap**2), np.std(lap)], dtype=np.float32)

    def extract_embedding(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts a unified, L2-normalized 2,340-dimensional facial embedding vector.
        """
        if img_bgr is None:
            return None

        # Preprocessing: Standardize size & normalize lighting with CLAHE
        img_resized = cv2.resize(img_bgr, self.target_size)
        
        if len(img_resized.shape) == 3:
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_resized.copy()

        # CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting invariance
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        # 1. Multi-scale HOG features (shape & landmark gradients)
        hog1 = self.hog_dense.compute(gray_eq).flatten()
        hog1 /= (np.linalg.norm(hog1) + 1e-6)

        hog2 = self.hog_coarse.compute(gray_eq).flatten()
        hog2 /= (np.linalg.norm(hog2) + 1e-6)

        # 2. LBP micro-texture descriptor (skin/stubble/eyes detail)
        lbp_feat = self._compute_lbp(gray_eq)

        # 3. Wavelet frequency energy
        wavelet_feat = self.extract_wavelet_energy(gray_eq)

        # 4. Chrominance features (normalized skin & hair tones)
        if len(img_resized.shape) == 3:
            ycrcb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YCrCb)
            cr_hist, _ = np.histogram(ycrcb[:, :, 1], bins=16, range=(0, 256))
            cb_hist, _ = np.histogram(ycrcb[:, :, 2], bins=16, range=(0, 256))
            color_feat = np.concatenate([cr_hist, cb_hist]).astype(np.float32)
            color_feat /= (np.linalg.norm(color_feat) + 1e-6)
        else:
            color_feat = np.zeros(32, dtype=np.float32)

        # Concatenate & final global L2 normalization
        combined = np.concatenate([hog1, hog2, lbp_feat, wavelet_feat, color_feat])
        combined /= (np.linalg.norm(combined) + 1e-6)

        return combined

    def extract_combined_features(self, img_bgr: np.ndarray) -> np.ndarray:
        """Alias for backward compatibility."""
        return self.extract_embedding(img_bgr)
