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


def _detect_by_saturation(image):

    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    saturation = cv2.GaussianBlur(saturation, (5, 5), 0)
    _, mask = cv2.threshold(saturation, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)

    image_area = float(image.shape[0] * image.shape[1])
    area_ratio = cv2.contourArea(largest) / image_area
    if not MIN_SHEET_AREA_RATIO <= area_ratio <= MAX_SHEET_AREA_RATIO:
        return None
    return _contour_to_quad(largest)


def _detect_by_edges(image):

    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(image.shape[0] * image.shape[1])

    candidates = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(contour) < image_area * MIN_SHEET_AREA_RATIO:
            break
        candidates.append(_contour_to_quad(contour))



    return max(candidates, key=lambda quad: _interior_brightness(gray, quad))
