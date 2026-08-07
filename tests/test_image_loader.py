"""Tests for Member 1 - src/image_loader.py."""

import cv2
import numpy as np
import pytest

from src.image_loader import (
    correct_perspective,
    crop_sheet,
    detect_sheet_boundary,
    load_image,
    resize_image,
    rotate_image,
)

REAL_SAMPLE = "data/signing_sheets/1.jpeg"


def make_synthetic_photo(width=800, height=600):
    """A tilted white 'sheet' on a dark desk background, like a phone photo."""
    photo = np.full((height, width, 3), 60, dtype=np.uint8)
    corners = np.array(
        [[150, 80], [660, 120], [630, 520], [120, 470]], np.int32)
    cv2.fillPoly(photo, [corners], (250, 250, 250))
    return photo


@pytest.fixture
def synthetic_photo():
    return make_synthetic_photo()


class TestLoadImage:
    def test_valid_jpg_loads(self, tmp_path, synthetic_photo):
        path = tmp_path / "sheet.jpg"
        cv2.imwrite(str(path), synthetic_photo)

        image = load_image(str(path))

        assert isinstance(image, np.ndarray)
        assert image.size > 0
        assert image.ndim == 3 and image.shape[2] == 3

    def test_real_sample_jpg_loads(self):
        image = load_image(REAL_SAMPLE)
        assert isinstance(image, np.ndarray)
        assert image.size > 0

    def test_valid_png_loads(self, tmp_path, synthetic_photo):
        path = tmp_path / "sheet.png"
        cv2.imwrite(str(path), synthetic_photo)

        image = load_image(str(path))

        assert isinstance(image, np.ndarray)
        assert image.size > 0
        assert image.ndim == 3 and image.shape[2] == 3

    def test_missing_file_shows_error(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_image("no/such/file.jpg")

    def test_non_image_file_shows_error(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("this is not an image")
        with pytest.raises(ValueError, match="not a readable image"):
            load_image(str(path))


class TestResizeImage:
    def test_resizing_preserves_proportions(self, synthetic_photo):
        resized = resize_image(synthetic_photo, max_dimension=400)

        original_ratio = synthetic_photo.shape[1] / synthetic_photo.shape[0]
        resized_ratio = resized.shape[1] / resized.shape[0]
        assert max(resized.shape[:2]) <= 400
        assert resized_ratio == pytest.approx(original_ratio, abs=0.02)

    def test_small_image_is_not_upscaled(self, synthetic_photo):
        resized = resize_image(synthetic_photo, max_dimension=5000)
        assert resized.shape == synthetic_photo.shape

    def test_invalid_max_dimension_rejected(self, synthetic_photo):
        with pytest.raises(ValueError):
            resize_image(synthetic_photo, max_dimension=0)


class TestRotateImage:
    def test_rotation_keeps_whole_image(self, synthetic_photo):
        rotated = rotate_image(synthetic_photo, 15)
        # Expanded canvas: both dimensions grow, nothing is cut off.
        assert rotated.shape[0] > synthetic_photo.shape[0]
        assert rotated.shape[1] > synthetic_photo.shape[1]

    def test_rotated_image_is_corrected(self, synthetic_photo):
        rotated = rotate_image(synthetic_photo, 12)

        corrected = crop_sheet(rotated)

        # The corrected sheet must be mostly white paper (background and
        # tilt removed), roughly matching the drawn sheet's proportions.
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        assert gray.mean() > 200
        ratio = corrected.shape[1] / corrected.shape[0]
        assert 0.9 < ratio < 1.7


class TestSheetDetectionAndPerspective:
    def test_boundary_has_four_ordered_corners(self, synthetic_photo):
        corners = detect_sheet_boundary(synthetic_photo)

        assert corners.shape == (4, 2)
        top_left, top_right, bottom_right, bottom_left = corners
        assert top_left[0] < top_right[0]
        assert bottom_left[0] < bottom_right[0]
        assert top_left[1] < bottom_left[1]
        assert top_right[1] < bottom_right[1]

    def test_perspective_correction_creates_usable_image(self, synthetic_photo):
        corners = detect_sheet_boundary(synthetic_photo)

        corrected = correct_perspective(synthetic_photo, corners)

        assert corrected.size > 0
        assert corrected.shape[0] > 100 and corrected.shape[1] > 100
        # Usable means the dark desk background is gone: near-white sheet.
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        assert gray.mean() > 200

    def test_invalid_corners_rejected(self, synthetic_photo):
        with pytest.raises(ValueError):
            correct_perspective(synthetic_photo, np.zeros((3, 2), np.float32))

    def test_real_sample_sheet_is_detected(self):
        image = resize_image(load_image(REAL_SAMPLE))

        corrected = crop_sheet(image)

        # The crop should remove background: smaller than the photo but
        # still large enough to contain the attendance table.
        assert corrected.size > 0
        assert corrected.shape[0] < image.shape[0] or corrected.shape[1] < image.shape[1]
        assert min(corrected.shape[:2]) > 300
