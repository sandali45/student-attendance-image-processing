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
