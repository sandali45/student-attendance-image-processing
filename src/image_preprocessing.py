"""

This module performs preprocessing on a corrected attendance-sheet image
before thresholding and table detection.

Pipeline:
1. Convert to grayscale
2. Apply Gaussian filter
3. Apply Median filter
4. Enhance contrast using CLAHE
5. Return the preprocessed image

Author: Member 2
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


# ---------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------

def _validate_image(image: np.ndarray) -> None:
    """
    Validate input image.

    Raises:
        TypeError
        ValueError
    """

    if image is None:
        raise ValueError("Image is None.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Input image is empty.")


def _validate_kernel(kernel: int | Tuple[int, int]) -> None:
    """
    Validate Gaussian/Median kernel size.
    """

    if isinstance(kernel, tuple):
        if len(kernel) != 2:
            raise ValueError("Kernel tuple must have exactly two values.")

        if kernel[0] <= 0 or kernel[1] <= 0:
            raise ValueError("Kernel dimensions must be positive.")

        if kernel[0] % 2 == 0 or kernel[1] % 2 == 0:
            raise ValueError("Kernel dimensions must be odd.")

    else:
        if kernel <= 0:
            raise ValueError("Kernel size must be positive.")

        if kernel % 2 == 0:
            raise ValueError("Kernel size must be odd.")


# ---------------------------------------------------------------------
# Image Processing Functions
# ---------------------------------------------------------------------

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image into grayscale.

    Parameters
    ----------
    image : np.ndarray

    Returns
    -------
    np.ndarray
        Grayscale image.
    """

    _validate_image(image)

    logger.info("Converting image to grayscale...")

    if len(image.shape) == 2:
        logger.info("Image already grayscale.")
        return image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    logger.info("Grayscale conversion completed.")

    return gray


def apply_gaussian_filter(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5)
) -> np.ndarray:
    """
    Reduce Gaussian noise.
    """

    _validate_image(image)
    _validate_kernel(kernel_size)

    logger.info("Applying Gaussian filter...")

    filtered = cv2.GaussianBlur(image, kernel_size, 0)

    logger.info("Gaussian filter completed.")

    return filtered


def apply_median_filter(
    image: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Remove salt-and-pepper noise.
    """

    _validate_image(image)
    _validate_kernel(kernel_size)

    logger.info("Applying Median filter...")

    filtered = cv2.medianBlur(image, kernel_size)

    logger.info("Median filter completed.")

    return filtered


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Enhance local contrast using CLAHE.
    """

    _validate_image(image)

    logger.info("Enhancing contrast using CLAHE...")

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    enhanced = clahe.apply(image)

    logger.info("Contrast enhancement completed.")

    return enhanced


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Complete preprocessing pipeline.

    Steps
    -----
    1. Grayscale
    2. Gaussian Blur
    3. Median Blur
    4. Contrast Enhancement

    Returns
    -------
    np.ndarray
    """

    logger.info("Starting preprocessing pipeline...")

    gray = convert_to_grayscale(image)

    gaussian = apply_gaussian_filter(gray)

    median = apply_median_filter(gaussian)

    enhanced = enhance_contrast(median)

    logger.info("Preprocessing pipeline completed.")

    return enhanced


# ---------------------------------------------------------------------
# Utility Function
# ---------------------------------------------------------------------

def save_image(image: np.ndarray, output_path: str | Path) -> None:
    """
    Save processed image.

    Parameters
    ----------
    image : np.ndarray

    output_path : str | Path
    """

    _validate_image(image)

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise IOError(f"Failed to save image: {output_path}")

    logger.info("Saved image -> %s", output_path)


# ---------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------

if __name__ == "__main__":

    IMAGE_PATH = "data/sample_sheet.png"

    logger.info("Loading image...")

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(
            f"Cannot open image: {IMAGE_PATH}"
        )

    processed = preprocess_image(image)

    save_image(
        processed,
        "output/grayscale_images/preprocessed.png"
    )

    logger.info("Processing finished successfully.")