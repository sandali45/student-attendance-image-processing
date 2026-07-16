import os
import sys

import cv2
import numpy as np


DEFAULT_MAX_DIMENSION = 1500


MIN_SHEET_AREA_RATIO = 0.20
MAX_SHEET_AREA_RATIO = 0.98

OUTPUT_DIR = os.path.join("output", "corrected_images")


def _validate_image(image):
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Expected a non-empty image (numpy array)")


def load_image(path):

    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(
            f"Image path must be a string, got {type(path).__name__}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"File exists but is not a readable image: {path}")
    return image


def resize_image(image, max_dimension=DEFAULT_MAX_DIMENSION):

    _validate_image(image)
    if max_dimension <= 0:
        raise ValueError(
            f"max_dimension must be positive, got {max_dimension}")

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return image

    scale = max_dimension / longest

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    new_size = (new_width, new_height)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def rotate_image(image, angle):

    _validate_image(image)

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # New bounding size after rotation.
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)

    # Shift so the rotated image stays centered in the new canvas.
    matrix[0, 2] += (new_width / 2.0) - center[0]
    matrix[1, 2] += (new_height / 2.0) - center[1]

    return cv2.warpAffine(image, matrix, (new_width, new_height),
                          borderValue=(255, 255, 255))


def detect_sheet_boundary(image):

    _validate_image(image)

    best_quad = _detect_by_saturation(image) if image.ndim == 3 else None
    if best_quad is None:
        best_quad = _detect_by_edges(image)

    if best_quad is None:
        height, width = image.shape[:2]
        best_quad = np.array([[0, 0], [width - 1, 0],
                              [width - 1, height - 1], [0, height - 1]])

    return _order_corners(np.asarray(best_quad, dtype=np.float32))
