import os
import sys

import cv2
import numpy as np


CORRECTED_DIR = os.path.join("output", "corrected_images")
BINARY_DIR = os.path.join("output", "binary_images")

TABLE_OUTPUT_DIR = os.path.join("output", "detected_tables")
ROWS_OUTPUT_DIR = os.path.join("output", "detected_rows")


def _validate_image(image):
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Expected a non-empty image (numpy array)")


# detect horizontal lines using a wide morphological kernel
def detect_horizontal_lines(binary_img, h_scale=30):
    _validate_image(binary_img)
    inverted = cv2.bitwise_not(binary_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (inverted.shape[1] // h_scale, 1))
    return cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)


# detect vertical lines using a tall morphological kernel
def detect_vertical_lines(binary_img, v_scale=30):
    _validate_image(binary_img)
    inverted = cv2.bitwise_not(binary_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, inverted.shape[0] // v_scale))
    return cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)


def combine_grid_lines(horizontal_lines, vertical_lines):
    _validate_image(horizontal_lines)
    _validate_image(vertical_lines)
    return cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)


# find cell/table contours from the combined grid image
# thresholds are ratio-based (relative to image size) so this works
# consistently across different sheet resolutions
def detect_table_boundary(grid_image, background_image, min_area_ratio=0.001,
                           min_width_ratio=0.03, min_height_ratio=0.01):
    _validate_image(grid_image)
    _validate_image(background_image)

    img_h, img_w = grid_image.shape[:2]
    img_area = img_h * img_w

    min_area = img_area * min_area_ratio
    min_width = img_w * min_width_ratio
    min_height = img_h * min_height_ratio

    contours, _ = cv2.findContours(grid_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(background_image.shape) == 2:
        preview = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR)
    else:
        preview = background_image.copy()

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > min_area and w > min_width and h > min_height:
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
            boxes.append((x, y, w, h))

    return boxes, preview


def get_table_body_box(boxes):
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])


# use the narrow leftmost column (No. column) to get one box per student row
def detect_row_boundaries(boxes, min_width_ratio=0.1, max_width_ratio=0.3 , expected_rows=6):
    table_body = get_table_body_box(boxes)
    if table_body is None:
        return []

    tx, ty, tw, th = table_body
    y_start = ty + int(th * 0.05)
    x_min = tx + int(tw * 0.65)
    x_max = tx + tw

    min_width = tw * min_width_ratio
    max_width = tw * max_width_ratio

    column_boxes = []
    for (x, y, w, h) in boxes:
        if (x_min <= x <= x_max and min_width <= w <= max_width
                and y > y_start and (x, y, w, h) != table_body):
            column_boxes.append((x, y, w, h))

    column_boxes.sort(key=lambda b: b[1])

    if expected_rows is not None and len(column_boxes) > expected_rows:
        column_boxes = column_boxes[:-expected_rows:]

    return [(y, y + h) for (x, y, w, h) in column_boxes]


def draw_row_boundaries(background_image, row_boundaries):
    _validate_image(background_image)
    vis = background_image.copy()
    width = vis.shape[1]

    for i, (top, bottom) in enumerate(row_boundaries):
        cv2.line(vis, (0, top), (width, top), (0, 0, 255), 2)
        cv2.putText(vis, f"R{i + 1}", (5, (top + bottom) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if row_boundaries:
        cv2.line(vis, (0, row_boundaries[-1][1]), (width, row_boundaries[-1][1]), (0, 0, 255), 2)

    return vis


def process_sheet_image(name, corrected_dir=CORRECTED_DIR, binary_dir=BINARY_DIR):
    binary_path = os.path.join(binary_dir, f"{name}_cleaned_binary.png")
    corrected_path = os.path.join(corrected_dir, f"{name}_corrected.png")

    binary_img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    corrected_img = cv2.imread(corrected_path, cv2.IMREAD_COLOR)

    if binary_img is None:
        raise FileNotFoundError(f"Binary image not found: {binary_path}")
    if corrected_img is None:
        raise FileNotFoundError(f"Corrected image not found: {corrected_path}")

    os.makedirs(TABLE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(ROWS_OUTPUT_DIR, exist_ok=True)

    print(f"[1/4] Detecting horizontal and vertical lines: {name}")
    h_lines = detect_horizontal_lines(binary_img)
    v_lines = detect_vertical_lines(binary_img)

    print("[2/4] Combining grid lines")
    grid = combine_grid_lines(h_lines, v_lines)

    print("[3/4] Detecting table boundary")
    boxes, table_preview = detect_table_boundary(grid, corrected_img)

    print("[4/4] Detecting row boundaries")
    row_boundaries = detect_row_boundaries(boxes)
    rows_preview = draw_row_boundaries(corrected_img, row_boundaries)

    cv2.imwrite(os.path.join(TABLE_OUTPUT_DIR, f"{name}_horizontal.png"), h_lines)
    cv2.imwrite(os.path.join(TABLE_OUTPUT_DIR, f"{name}_vertical.png"), v_lines)
    cv2.imwrite(os.path.join(TABLE_OUTPUT_DIR, f"{name}_table.png"), table_preview)
    cv2.imwrite(os.path.join(ROWS_OUTPUT_DIR, f"{name}_rows.png"), rows_preview)

    print(f"Sheet {name}: {len(boxes)} boxes, {len(row_boundaries)} rows detected")
    print(f"Saved outputs to {TABLE_OUTPUT_DIR} and {ROWS_OUTPUT_DIR}")

    return {"boxes": boxes, "row_boundaries": row_boundaries}


def run(sheet_names, corrected_dir=CORRECTED_DIR, binary_dir=BINARY_DIR):
    results = {}
    for name in sheet_names:
        try:
            results[name] = process_sheet_image(name, corrected_dir, binary_dir)
        except FileNotFoundError as e:
            print(f"Skipping sheet {name}: {e}")
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sheet_names = sys.argv[1:]
    else:
        sheet_names = ["1", "2", "3", "4", "5"]

    run(sheet_names)
