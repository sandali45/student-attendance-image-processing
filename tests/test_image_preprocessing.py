import os
import tempfile

import cv2
import numpy as np
import pytest

from src.image_preprocessing import (
    convert_to_grayscale,
    apply_gaussian_filter,
    apply_median_filter,
    enhance_contrast,
    preprocess_image,
    save_image,
)


# ---------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------

@pytest.fixture
def sample_color_image():
    """
    Create a random RGB image.
    """
    return np.random.randint(
        0,
        256,
        (200, 300, 3),
        dtype=np.uint8
    )


@pytest.fixture
def sample_gray_image():
    """
    Create a random grayscale image.
    """
    return np.random.randint(
        0,
        256,
        (200, 300),
        dtype=np.uint8
    )


# ---------------------------------------------------------
# Grayscale Tests
# ---------------------------------------------------------

def test_convert_to_grayscale(sample_color_image):

    gray = convert_to_grayscale(sample_color_image)

    assert len(gray.shape) == 2
    assert gray.shape == sample_color_image.shape[:2]


def test_grayscale_input_returns_same_shape(sample_gray_image):

    gray = convert_to_grayscale(sample_gray_image)

    assert gray.shape == sample_gray_image.shape


# ---------------------------------------------------------
# Gaussian Filter Tests
# ---------------------------------------------------------

def test_gaussian_filter_preserves_size(sample_gray_image):

    result = apply_gaussian_filter(sample_gray_image)

    assert result.shape == sample_gray_image.shape


def test_gaussian_invalid_kernel(sample_gray_image):

    with pytest.raises(ValueError):
        apply_gaussian_filter(sample_gray_image, (4, 4))


# ---------------------------------------------------------
# Median Filter Tests
# ---------------------------------------------------------

def test_median_filter_preserves_size(sample_gray_image):

    result = apply_median_filter(sample_gray_image)

    assert result.shape == sample_gray_image.shape


def test_median_invalid_kernel(sample_gray_image):

    with pytest.raises(ValueError):
        apply_median_filter(sample_gray_image, 4)


# ---------------------------------------------------------
# CLAHE Tests
# ---------------------------------------------------------

def test_contrast_enhancement(sample_gray_image):

    result = enhance_contrast(sample_gray_image)

    assert result.shape == sample_gray_image.shape


# ---------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------

def test_preprocess_pipeline(sample_color_image):

    processed = preprocess_image(sample_color_image)

    assert len(processed.shape) == 2

    assert processed.shape == sample_color_image.shape[:2]


# ---------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------

def test_none_image():

    with pytest.raises(ValueError):
        convert_to_grayscale(None)


def test_empty_image():

    empty = np.array([], dtype=np.uint8)

    with pytest.raises(ValueError):
        convert_to_grayscale(empty)


# ---------------------------------------------------------
# Save Image Test
# ---------------------------------------------------------

def test_save_image(sample_gray_image):

    with tempfile.TemporaryDirectory() as temp_dir:

        output = os.path.join(temp_dir, "test.png")

        save_image(sample_gray_image, output)

        assert os.path.exists(output)

        loaded = cv2.imread(output, cv2.IMREAD_GRAYSCALE)

        assert loaded is not None

        assert loaded.shape == sample_gray_image.shape


# ---------------------------------------------------------
# Noise Reduction Test
# ---------------------------------------------------------

def test_noise_reduction(sample_gray_image):

    noisy = sample_gray_image.copy()

    noise = np.random.randint(
        0,
        30,
        noisy.shape,
        dtype=np.uint8
    )

    noisy = cv2.add(noisy, noise)

    gaussian = apply_gaussian_filter(noisy)

    median = apply_median_filter(gaussian)

    assert median.shape == noisy.shape

    assert median.dtype == noisy.dtype