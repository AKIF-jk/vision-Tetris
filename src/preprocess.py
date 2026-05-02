# =============================================================================
# src/preprocess.py — Task 3: Image Preprocessing Module
#
# Input:  Raw PNG/JPG screenshot (any resolution)
# Output: Preprocessed binary image (numpy array) + debug images
# =============================================================================

import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_image(path: str, verbose: bool = False) -> np.ndarray:
    """Load image from disk. Raises FileNotFoundError if path invalid."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if verbose:
        print(f"  [preprocess] Loaded image: {path} | shape: {img.shape}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_blur(gray: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur to reduce high-frequency noise."""
    return cv2.GaussianBlur(gray, config.GAUSSIAN_BLUR_KERNEL, 0)


def apply_threshold(blurred: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Binarize the image.
    method='otsu'  — automatic threshold (recommended for varied screenshots)
    method='fixed' — use config.BINARY_THRESHOLD (faster, less adaptive)
    """
    if method == "otsu":
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blurred, config.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary


def preprocess(img: np.ndarray,
               save_debug: bool = False,
               debug_prefix: str = "frame") -> np.ndarray:
    """
    Full preprocessing pipeline.

    Args:
        img          : BGR image (loaded via load_image or cv2.imread)
        save_debug   : If True, saves intermediate images to config.DEBUG_DIR
        debug_prefix : Filename prefix for debug images

    Returns:
        binary_img   : Binarized numpy array (uint8, values 0 or 255)
    """
    gray     = to_grayscale(img)
    blurred  = apply_blur(gray)
    binary   = apply_threshold(blurred, method="otsu")

    if save_debug:
        os.makedirs(config.DEBUG_DIR, exist_ok=True)
        cv2.imwrite(f"{config.DEBUG_DIR}/{debug_prefix}_1_gray.jpg",    gray)
        cv2.imwrite(f"{config.DEBUG_DIR}/{debug_prefix}_2_blurred.jpg", blurred)
        cv2.imwrite(f"{config.DEBUG_DIR}/{debug_prefix}_3_binary.jpg",  binary)

    return binary


# =============================================================================
# Quick test — run this file directly to verify preprocessing works
# =============================================================================
if __name__ == "__main__":
    import glob
    screenshots = glob.glob(f"{config.INPUT_DIR}/*.jpg") + glob.glob(f"{config.INPUT_DIR}/*.png")
    if not screenshots:
        print("No screenshots found in", config.INPUT_DIR)
    else:
        test_path = screenshots[0]
        img    = load_image(test_path, verbose=True)
        binary = preprocess(img, save_debug=True, debug_prefix="test")
        print(f"  [preprocess] Binary image shape: {binary.shape}")
        print(f"  [preprocess] Non-zero pixels (filled): {np.count_nonzero(binary)}")
        print("  [preprocess] Done. Check output/debug/ for results.")
