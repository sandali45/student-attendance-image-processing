"""
Attendance detection module.

Member 6 responsibility:
Determine whether each extracted student signature region
contains a signature (Present) or is empty (Absent).

Input:
    Cropped signature region produced by Member 5.

Output:
    Row number, present/absent decision, presence score,
    ink ratio and contour count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# Starting value only.
# This MUST be tuned after testing the real extracted signature boxes.
DEFAULT_PRESENCE_THRESHOLD = 0.020

# Connected components smaller than this are considered noise.
DEFAULT_MIN_COMPONENT_AREA = 8

# Percentage of the image edges ignored to remove table borders.
DEFAULT_BORDER_RATIO = 0.06


def _validate_image(image: np.ndarray) -> None:
    """Check that the supplied image is usable."""

    if image is None:
        raise ValueError("Signature image cannot be None.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Signature image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Signature image cannot be empty.")

    if image.ndim not in (2, 3):
        raise ValueError("Signature image must be grayscale or colour.")


def _create_ink_mask(image: np.ndarray) -> np.ndarray:
    """
    Convert a signature region into a binary ink mask.

    White pixels (255) = possible handwriting/ink.
    Black pixels (0)   = background.

    Both dark ink and coloured ink are considered because
    students may use different coloured pens.
    """

    _validate_image(image)

    # Colour image
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect dark objects such as black/blue handwriting.
        _, dark_mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        # Also detect strongly coloured pen strokes.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        colour_mask = np.where(
            (saturation > 45) & (value < 250),
            255,
            0,
        ).astype(np.uint8)

        ink_mask = cv2.bitwise_or(dark_mask, colour_mask)

    # Already grayscale
    else:
        gray = image

        _, ink_mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

    return ink_mask


def _remove_border(
    binary_mask: np.ndarray,
    border_ratio: float = DEFAULT_BORDER_RATIO,
) -> np.ndarray:
    """
    Remove pixels close to the edge of the cropped region.

    Member 5 should already remove most table borders, but this
    provides additional protection against border lines being
    classified as signatures.
    """

    if not 0 <= border_ratio < 0.5:
        raise ValueError("border_ratio must be between 0 and 0.5.")

    cleaned = binary_mask.copy()

    height, width = cleaned.shape

    border_y = max(2, int(height * border_ratio))
    border_x = max(2, int(width * border_ratio))

    cleaned[:border_y, :] = 0
    cleaned[-border_y:, :] = 0
    cleaned[:, :border_x] = 0
    cleaned[:, -border_x:] = 0

    return cleaned


def remove_small_noise(
    binary_mask: np.ndarray,
    min_area: int = DEFAULT_MIN_COMPONENT_AREA,
) -> np.ndarray:
    """
    Remove tiny connected components.

    Small isolated dots may come from camera noise, dirt or printing.
    These should not cause a student to be marked Present.
    """

    if min_area < 1:
        raise ValueError("min_area must be at least 1.")

    if binary_mask is None or binary_mask.size == 0:
        raise ValueError("Binary mask cannot be empty.")

    # Ensure binary format.
    binary_mask = np.where(binary_mask > 0, 255, 0).astype(np.uint8)

    number_of_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(binary_mask)

    # Label 0 is the background, therefore start at 1.
    for label in range(1, number_of_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def prepare_signature_mask(
    signature_region: np.ndarray,
    min_area: int = DEFAULT_MIN_COMPONENT_AREA,
    border_ratio: float = DEFAULT_BORDER_RATIO,
) -> np.ndarray:
    """
    Prepare an extracted signature box for attendance detection.
    """

    ink_mask = _create_ink_mask(signature_region)

    without_border = _remove_border(
        ink_mask,
        border_ratio=border_ratio,
    )

    cleaned = remove_small_noise(
        without_border,
        min_area=min_area,
    )

    return cleaned


def calculate_ink_ratio(
    signature_region: np.ndarray,
    min_area: int = DEFAULT_MIN_COMPONENT_AREA,
    border_ratio: float = DEFAULT_BORDER_RATIO,
) -> float:
    """
    Calculate the percentage of the usable signature region
    occupied by detected ink.

    Example:
        0.084 means approximately 8.4% of the region contains ink.
    """

    cleaned = prepare_signature_mask(
        signature_region,
        min_area=min_area,
        border_ratio=border_ratio,
    )

    height, width = cleaned.shape

    border_y = max(2, int(height * border_ratio))
    border_x = max(2, int(width * border_ratio))

    usable_height = max(1, height - (2 * border_y))
    usable_width = max(1, width - (2 * border_x))

    usable_area = usable_height * usable_width

    ink_pixels = cv2.countNonZero(cleaned)

    return float(ink_pixels / usable_area)


def count_contours(
    signature_region: np.ndarray,
    min_area: int = DEFAULT_MIN_COMPONENT_AREA,
    border_ratio: float = DEFAULT_BORDER_RATIO,
) -> int:
    """
    Count meaningful ink components in a signature region.
    """

    cleaned = prepare_signature_mask(
        signature_region,
        min_area=min_area,
        border_ratio=border_ratio,
    )

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    meaningful_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= min_area
    ]

    return len(meaningful_contours)


def calculate_presence_score(
    signature_region: np.ndarray,
    min_area: int = DEFAULT_MIN_COMPONENT_AREA,
    border_ratio: float = DEFAULT_BORDER_RATIO,
) -> float:
    """
    Calculate the signature-presence score.

    Currently the score is the cleaned ink ratio.

    Keeping the score simple makes the attendance decision
    explainable and easy to tune using the supplied signing sheets.
    """

    return calculate_ink_ratio(
        signature_region,
        min_area=min_area,
        border_ratio=border_ratio,
    )


def classify_present_absent(
    signature_region: np.ndarray,
    row_number: int,
    threshold: float = DEFAULT_PRESENCE_THRESHOLD,
    min_area: int = DEFAULT_MIN_COMPONENT_AREA,
    border_ratio: float = DEFAULT_BORDER_RATIO,
) -> dict:
    """
    Classify one student's signature box as Present or Absent.

    Returns:
        {
            "row_number": 1,
            "present": True,
            "status": "Present",
            "score": 0.084,
            "ink_ratio": 0.084,
            "contour_count": 3,
            "threshold": 0.020
        }
    """

    if row_number < 1:
        raise ValueError("row_number must be 1 or greater.")

    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")

    ink_ratio = calculate_ink_ratio(
        signature_region,
        min_area=min_area,
        border_ratio=border_ratio,
    )

    contour_count = count_contours(
        signature_region,
        min_area=min_area,
        border_ratio=border_ratio,
    )

    score = calculate_presence_score(
        signature_region,
        min_area=min_area,
        border_ratio=border_ratio,
    )

    # Presence requires enough detected ink AND at least one
    # meaningful connected ink component.
    present = score >= threshold and contour_count > 0

    return {
        "row_number": row_number,
        "present": present,
        "status": "Present" if present else "Absent",
        "score": round(score, 4),
        "ink_ratio": round(ink_ratio, 4),
        "contour_count": contour_count,
        "threshold": threshold,
    }


def classify_attendance(
    signature_regions: list,
    threshold: float = DEFAULT_PRESENCE_THRESHOLD,
) -> list[dict]:
    """
    Classify all signature regions received from Member 5.

    Preferred Member 5 format:

        [
            {
                "row_number": 1,
                "image": numpy_array
            },
            {
                "row_number": 2,
                "image": numpy_array
            }
        ]
    """

    if not isinstance(signature_regions, list):
        raise TypeError("signature_regions must be a list.")

    results = []

    for position, region in enumerate(signature_regions, start=1):

        if isinstance(region, dict):
            if "image" not in region:
                raise ValueError(
                    f"Signature region {position} does not contain 'image'."
                )

            row_number = region.get("row_number", position)
            image = region["image"]

        else:
            # Also support a simple list of NumPy image arrays.
            row_number = position
            image = region

        result = classify_present_absent(
            signature_region=image,
            row_number=row_number,
            threshold=threshold,
        )

        results.append(result)

    return results


def analyse_signature_folder(
    folder_path: str,
    threshold: float = DEFAULT_PRESENCE_THRESHOLD,
) -> list[dict]:
    """
    Temporary helper for Member 6 testing.

    Reads cropped signature images from a folder such as:

        output/signature_regions/2019-07-12/

    Files should ideally be named:
        row_01.png
        row_02.png
        ...
    """

    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(
            f"Signature folder does not exist: {folder}"
        )

    valid_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
    }

    files = sorted(
        file
        for file in folder.iterdir()
        if file.suffix.lower() in valid_extensions
    )

    if not files:
        raise ValueError(
            f"No signature images found inside: {folder}"
        )

    results = []

    for row_number, file_path in enumerate(files, start=1):

        image = cv2.imread(str(file_path))

        if image is None:
            raise ValueError(
                f"Could not read image: {file_path}"
            )

        result = classify_present_absent(
            signature_region=image,
            row_number=row_number,
            threshold=threshold,
        )

        result["filename"] = file_path.name

        results.append(result)

    return results


def save_results(
    results: list[dict],
    output_path: str,
) -> None:
    """Save Member 6 results as JSON for testing/integration."""

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )


def _print_results(results: list[dict]) -> None:
    """Print attendance results clearly in the terminal."""

    print("\nAttendance Detection Results")
    print("-" * 55)

    for result in results:
        print(
            f"Row {result['row_number']:02d} | "
            f"{result['status']:<7} | "
            f"Score: {result['score']:.4f} | "
            f"Contours: {result['contour_count']}"
        )

    print("-" * 55)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Detect Present/Absent from extracted signature boxes."
    )

    parser.add_argument(
        "folder",
        help="Folder containing cropped signature-box images.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_PRESENCE_THRESHOLD,
        help="Presence threshold. Default: 0.020",
    )

    parser.add_argument(
        "--output",
        default="output/attendance_results/member6_results.json",
        help="JSON output file.",
    )

    args = parser.parse_args()

    attendance_results = analyse_signature_folder(
        folder_path=args.folder,
        threshold=args.threshold,
    )

    _print_results(attendance_results)

    save_results(
        attendance_results,
        args.output,
    )

    print(f"\nResults saved to: {args.output}")