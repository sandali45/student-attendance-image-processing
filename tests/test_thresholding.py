# tests/test_thresholding.py
# Basic tests for the binarization functions.
# run with: pytest tests/test_thresholding.py -v

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from thresholding import (
    global_threshold,
    otsu_threshold,
    adaptive_threshold,
    morphological_cleaning,
    select_best_binary_image,
)


def is_binary(img):
    # helper to check the image only has 0s and 255s in it
    vals = np.unique(img)
    return all(v == 0 or v == 255 for v in vals)


# a simple test image - dark square in the middle of a light background
def make_simple_image():
    img = np.full((100, 100), 220, dtype=np.uint8)
    img[30:70, 30:70] = 30
    return img


# image with a lighting gradient (like a shadow) + two squares,
# one on the bright side and one on the dark side
def make_uneven_image():
    h, w = 200, 200
    grad = np.tile(np.linspace(240, 90, w), (h, 1)).astype(np.uint8)
    img = grad.copy()
    img[40:70, 20:50] -= 60
    img[130:160, 150:180] -= 60
    return img


def test_output_only_0_and_255():
    img = make_simple_image()
    g = global_threshold(img)
    o = otsu_threshold(img)
    a = adaptive_threshold(img)
    c = morphological_cleaning(o)

    assert is_binary(g)
    assert is_binary(o)
    assert is_binary(a)
    assert is_binary(c)


def test_global_threshold_works():
    img = make_simple_image()
    result = global_threshold(img, thresh_value=127)

    # dark square should turn black, background should turn white
    assert result[50, 50] == 0
    assert result[5, 5] == 255


def test_otsu_threshold_works():
    img = make_simple_image()
    result = otsu_threshold(img)

    assert is_binary(result)
    assert result[50, 50] == 0
    assert result[5, 5] == 255


def test_adaptive_threshold_handles_uneven_lighting():
    img = make_uneven_image()
    result = adaptive_threshold(img, block_size=31, c=5)

    bright_side = result[40:70, 20:50]
    dark_side = result[130:160, 150:180]

    # both squares should be picked up as ink even though the
    # background lighting is totally different on each side
    assert np.mean(bright_side == 0) > 0.6
    assert np.mean(dark_side == 0) > 0.6


def test_blank_image_does_not_crash():
    blank = np.full((80, 80), 255, dtype=np.uint8)

    g = global_threshold(blank)
    o = otsu_threshold(blank)
    a = adaptive_threshold(blank)
    c = morphological_cleaning(o)

    # just making sure none of these crashed and the shape is fine
    assert g.shape == blank.shape
    assert o.shape == blank.shape
    assert a.shape == blank.shape
    assert c.shape == blank.shape

    # a blank page should mostly stay white
    assert np.mean(g == 255) > 0.9


def test_none_image_does_not_crash_program():
    # should raise a normal error, not some random cv2 crash
    with pytest.raises(ValueError):
        global_threshold(None)


def test_morphological_cleaning_removes_small_noise():
    img = np.full((100, 100), 255, dtype=np.uint8)
    img[40:60, 40:60] = 0  # a real signature-ish blob

    # sprinkle in some random single-pixel noise
    rng = np.random.default_rng(0)
    for i in range(40):
        y, x = rng.integers(0, 100, size=2)
        img[y, x] = 0

    cleaned = morphological_cleaning(img)

    # the real blob should mostly survive
    assert np.mean(cleaned[40:60, 40:60] == 0) > 0.8

    # but overall there should be less black pixels than before
    # (since the noise got cleaned up)
    assert np.sum(cleaned == 0) <= np.sum(img == 0)


def test_select_best_binary_image_picks_something_reasonable():
    all_black = np.zeros((50, 50), dtype=np.uint8)
    all_white = np.full((50, 50), 255, dtype=np.uint8)

    reasonable = np.full((50, 50), 255, dtype=np.uint8)
    reasonable[20:30, 20:30] = 0  # small bit of ink, like a real signature

    candidates = {
        "all_black": all_black,
        "all_white": all_white,
        "reasonable": reasonable,
    }

    best_name, best_img, scores = select_best_binary_image(candidates)
    assert best_name == "reasonable"


def test_select_best_binary_image_empty_dict_raises_error():
    with pytest.raises(ValueError):
        select_best_binary_image({})
