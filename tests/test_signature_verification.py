"""
tests/test_signature_verification.py
Member 10 -- Signature comparison and mismatch detection tests.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.signature_verification import (
    normalize_signature,
    resize_signature,
    calculate_ssim_score,
    calculate_orb_score,
    compare_with_reference_signatures,
    classify_match_or_mismatch,
    is_signature_blank,
)


def make_signature_image(seed=1, size=(200, 400)):
    img = np.full((size[0], size[1], 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    x, y = 20, size[0] // 2
    points = [(x, y)]
    for _ in range(15):
        x += int(rng.integers(15, 30))
        y += int(rng.integers(-40, 40))
        y = max(10, min(size[0] - 10, y))
        points.append((x, y))
        if x > size[1] - 20:
            break
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], (0, 0, 0), 3)
    return img


def make_blank_image(size=(200, 400)):
    return np.full((size[0], size[1], 3), 255, dtype=np.uint8)


@pytest.fixture
def reference_dir(tmp_path):
    ref_dir = tmp_path / "001"
    ref_dir.mkdir()
    for i in range(3):
        cv2.imwrite(str(ref_dir / f"ref_{i}.png"), make_signature_image(seed=100))
    return str(ref_dir)


def test_same_signature_receives_a_high_score(reference_dir):
    candidate = make_signature_image(seed=100)
    result = compare_with_reference_signatures(candidate, reference_dir)
    assert result["best_score"] > 0.6
    assert result["decision"] == "Matching"


def test_different_signature_receives_a_lower_score(reference_dir):
    genuine = compare_with_reference_signatures(make_signature_image(seed=100), reference_dir)
    forged = compare_with_reference_signatures(make_signature_image(seed=999), reference_dir)
    assert forged["best_score"] < genuine["best_score"]


def test_different_image_sizes_are_normalized():
    img_large = make_signature_image(seed=5, size=(200, 400))
    img_small = make_signature_image(seed=5, size=(120, 260))

    norm_large = resize_signature(normalize_signature(img_large))
    norm_small = resize_signature(normalize_signature(img_small))

    assert norm_large.shape == norm_small.shape
    score = calculate_ssim_score(norm_large, norm_small)
    assert score > 0.5


def test_blank_signature_is_rejected(reference_dir):
    result = compare_with_reference_signatures(make_blank_image(), reference_dir)
    assert result["student_signature_blank"] == True
    assert result["decision"] == "No Signature (Blank)"
    assert result["reference_count"] == 0


def test_missing_reference_signatures_are_handled(tmp_path):
    candidate = make_signature_image(seed=1)
    nonexistent_dir = str(tmp_path / "no_such_student")
    result = compare_with_reference_signatures(candidate, nonexistent_dir)
    assert result["decision"] == "Cannot Verify - No Reference Signatures"
    assert result["reference_count"] == 0
    assert result["best_reference"] is None


def test_threshold_decision_is_correct():
    assert classify_match_or_mismatch(0.80, threshold=0.55) == "Matching"
    assert classify_match_or_mismatch(0.40, threshold=0.55) == "Not Matching"
    assert classify_match_or_mismatch(0.55, threshold=0.55) == "Matching"


def test_calculate_orb_score_handles_featureless_images():
    tiny = np.zeros((150, 300), dtype=np.uint8)
    tiny[70:75, 140:160] = 255
    score = calculate_orb_score(tiny, tiny)
    assert 0.0 <= score <= 1.0


def test_normalize_signature_rejects_empty_image():
    empty = np.array([])
    with pytest.raises(ValueError):
        normalize_signature(empty)


def test_resize_signature_rejects_zero_dimension():
    bad = np.zeros((0, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        resize_signature(bad)


def test_is_signature_blank_detects_blank():
    blank = make_blank_image()
    assert is_signature_blank(blank) == True

    signed = make_signature_image(seed=1)
    assert is_signature_blank(signed) == False