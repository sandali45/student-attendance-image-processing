"""
src/signature_verification.py
Member 10 -- Signature comparison and mismatch detection.

Techniques used:
    - Image normalization (grayscale + binarization)
    - SSIM (Structural Similarity Index) for pixel-level comparison
    - ORB (Oriented FAST and Rotated BRIEF) for feature matching
    - Combined scoring with weighted average
"""

import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def normalize_signature(image):
    """Convert image to clean binary form for comparison."""
    if image is None or image.size == 0:
        raise ValueError("Empty image provided")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    return binary


def resize_signature(image, target_size=(200, 400)):
    """Resize to standard dimensions so all signatures are comparable."""
    if image is None:
        raise ValueError("Empty image")

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Zero dimension")

    interp = cv2.INTER_AREA if h > target_size[0] or w > target_size[1] else cv2.INTER_CUBIC
    return cv2.resize(image, (target_size[1], target_size[0]), interpolation=interp)


def calculate_ssim_score(image1, image2):
    """SSIM comparison after normalization and fixed resizing."""
    if image1 is None or image2 is None:
        return 0.0

    norm1 = resize_signature(normalize_signature(image1))
    norm2 = resize_signature(normalize_signature(image2))

    score, _ = ssim(
        norm1,
        norm2,
        full=True,
        data_range=255
    )

    return float(score)


def calculate_orb_score(image1, image2):
    """ORB feature matching: robust to slight rotation/scale changes."""
    if image1 is None or image2 is None:
        return 0.0

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1.copy()
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape) == 3 else image2.copy()

    orb = cv2.ORB_create(
        nfeatures=1000,
        scaleFactor=1.2,
        edgeThreshold=5
    )
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) == 0:
        return 0.0

    max_possible = min(len(kp1), len(kp2))
    if max_possible == 0:
        return 0.0

    return min(len(matches) / max_possible, 1.0)


def classify_match_or_mismatch(score, threshold=0.55):
    """Classify combined score. 0.55 balances false positives and false negatives."""
    return "Matching" if score >= threshold else "Not Matching"


def is_signature_blank(image, blank_threshold=0.02):
    """Check if signature box is empty (less than 2% ink)."""
    if image is None or image.size == 0:
        return True

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    ink_ratio = np.sum(gray < 240) / gray.size
    return ink_ratio < blank_threshold


def compare_with_reference_signatures(candidate_path, reference_dir, threshold=0.55):
    """Compare candidate against all references. Returns best match score and details."""
    result = {
        "best_score": 0.0,
        "best_reference": None,
        "decision": "Cannot Verify - No Reference Signatures",
        "results": [],
        "reference_count": 0,
        "student_signature_blank": False,
    }

    candidate = cv2.imread(candidate_path) if isinstance(candidate_path, str) else candidate_path
    if candidate is None:
        result["decision"] = "Error - Could not load image"
        return result

    if is_signature_blank(candidate):
        result["student_signature_blank"] = True
        result["decision"] = "No Signature (Blank)"
        return result

    if not os.path.isdir(reference_dir):
        return result

    ref_files = [f for f in sorted(os.listdir(reference_dir))
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    result["reference_count"] = len(ref_files)
    if len(ref_files) == 0:
        return result

    all_scores = []
    for ref_file in ref_files:
        ref_img = cv2.imread(os.path.join(reference_dir, ref_file))
        if ref_img is None:
            continue

        ssim_val = calculate_ssim_score(candidate, ref_img)
        orb_val = calculate_orb_score(candidate, ref_img)
        combined = (ssim_val * 0.6) + (orb_val * 0.4)

        all_scores.append({
            "reference_file": ref_file,
            "ssim_score": round(ssim_val, 4),
            "orb_score": round(orb_val, 4),
            "combined_score": round(combined, 4),
        })

        if combined > result["best_score"]:
            result["best_score"] = combined
            result["best_reference"] = ref_file

    result["results"] = all_scores
    result["best_score"] = round(result["best_score"], 4)
    result["decision"] = classify_match_or_mismatch(result["best_score"], threshold)

    return result