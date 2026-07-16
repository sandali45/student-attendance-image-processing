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

    if not candidates:
     return None

    return max(candidates, key=lambda quad: _interior_brightness(gray, quad))


def correct_perspective(image, corners):

    _validate_image(image)
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape != (4, 2):
        raise ValueError(
            f"Expected 4 corner points (4, 2), got shape {corners.shape}")

    top_left, top_right, bottom_right, bottom_left = corners

    width = int(max(np.linalg.norm(bottom_right - bottom_left),
                    np.linalg.norm(top_right - top_left)))
    height = int(max(np.linalg.norm(top_right - bottom_right),
                     np.linalg.norm(top_left - bottom_left)))
    if width < 1 or height < 1:
        raise ValueError("Corner points collapse to a degenerate rectangle")

    destination = np.array([[0, 0], [width - 1, 0],
                            [width - 1, height - 1], [0, height - 1]],
                           dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, matrix, (width, height))

def crop_sheet(image):

    corners = detect_sheet_boundary(image)
    return correct_perspective(image, corners)


def process_sheet_image(path, output_dir=OUTPUT_DIR):

    name = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/4] Loading image: {path}")
    image = load_image(path)

    print(f"[2/4] Resizing from {image.shape[1]}x{image.shape[0]}")
    image = resize_image(image)

    print("[3/4] Detecting sheet boundary")
    corners = detect_sheet_boundary(image)
    boundary_preview = image.copy()
    cv2.polylines(boundary_preview, [
                  corners.astype(np.int32)], True, (0, 255, 0), 3)

    print("[4/4] Correcting perspective and cropping")
    corrected = correct_perspective(image, corners)

    cv2.imwrite(os.path.join(output_dir, f"{name}_original.png"), image)
    cv2.imwrite(os.path.join(
        output_dir, f"{name}_boundary.png"), boundary_preview)
    cv2.imwrite(os.path.join(output_dir, f"{name}_corrected.png"), corrected)
    print(f"Saved original, boundary and corrected images to {output_dir}")

    return corrected


def _order_corners(corners):

    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = corners.sum(axis=1)
    diffs = np.diff(corners, axis=1).ravel()  # y - x
    ordered[0] = corners[np.argmin(sums)]   # top-left: smallest x + y
    ordered[2] = corners[np.argmax(sums)]   # bottom-right: largest x + y
    ordered[1] = corners[np.argmin(diffs)]  # top-right: smallest y - x
    ordered[3] = corners[np.argmax(diffs)]  # bottom-left: largest y - x
    return ordered


def _contour_to_quad(contour):

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        return approx.reshape(4, 2)
    return cv2.boxPoints(cv2.minAreaRect(contour))


def _interior_brightness(gray, quad):

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(quad, dtype=np.int32)], 255)
    return cv2.mean(gray, mask=mask)[0]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.image_loader <image_path>")
        sys.exit(1)
    process_sheet_image(sys.argv[1])
