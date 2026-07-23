import os

import cv2
import numpy as np

# reuse Member 4's helper if it's importable, otherwise fall back to a
# tiny local version so this module still works on its own
try:
    from table_detection import get_table_body_box
except Exception:  # pragma: no cover - only hit if Member 4's file is missing
    def get_table_body_box(boxes):
        return max(boxes, key=lambda b: b[2] * b[3]) if boxes else None


OUTPUT_DIR = os.path.join("output", "signature_regions")


def _validate_image(image):
    """Make sure we actually got an image and not None/empty. Same idea
    as the check the other members use so nothing crashes deep inside
    cv2 when an upstream step returns nothing."""
    if image is None:
        raise ValueError("image is None")
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image is empty")
    return image


def extract_student_rows(row_boundaries, min_height=4):
    """Clean up Member 4's row_boundaries into a tidy list of student
    rows. We drop anything with a silly height (0 or negative, usually a
    detection glitch) and sort them top-to-bottom so row 1 really is the
    top student - that ordering is what lets Member 7 line the rows up
    with the students in info.xml."""
    if row_boundaries is None:
        raise ValueError("row_boundaries is None")

    rows = []
    for rb in row_boundaries:
        top, bottom = int(rb[0]), int(rb[1])
        if bottom - top >= min_height:
            rows.append((top, bottom))

    rows.sort(key=lambda r: r[0])
    return rows


def detect_signature_column(boxes):
    """Figure out the x-range (left, right) of the signature column.

    On these sheets the signature column is the rightmost one, so we look
    at the table body box, then keep the narrow-ish cells sitting on the
    right-hand side of it. The common left/right edge of those cells is
    the signature column. If we can't find any (table detection was
    rough), we just fall back to the rightmost third of the table."""
    if not boxes:
        raise ValueError("no boxes given; cannot locate the signature column")

    table_body = get_table_body_box(boxes)
    tx, ty, tw, th = table_body

    x_min = tx + int(tw * 0.6)
    x_max = tx + tw

    candidates = [
        (x, y, w, h) for (x, y, w, h) in boxes
        if (x, y, w, h) != table_body
        and x >= x_min and (x + w) <= x_max + 2
        and 0.08 * tw <= w <= 0.4 * tw
    ]

    if candidates:
        left = int(np.median([b[0] for b in candidates]))
        right = int(np.median([b[0] + b[2] for b in candidates]))
    else:
        left, right = tx + int(tw * 0.68), tx + tw

    if right <= left:  # last-resort guard
        left, right = tx + int(tw * 0.68), tx + tw

    return left, right


def add_inner_margin(rect, margin, image_shape=None):
    """Shrink a rectangle inward by `margin` pixels on every side.

    rect is (left, top, right, bottom). We do this so the crop lands
    *inside* the cell and skips the black grid lines around it. If
    image_shape is given we clamp to it so we never go out of bounds, and
    if the margin would collapse the box to nothing we just return the
    original (clamped) rect instead."""
    left, top, right, bottom = [int(v) for v in rect]

    shrunk = (left + margin, top + margin, right - margin, bottom - margin)

    def clamp(r):
        l, t, rr, bb = r
        if image_shape is not None:
            h, w = image_shape[:2]
            l = max(0, min(l, w))
            rr = max(0, min(rr, w))
            t = max(0, min(t, h))
            bb = max(0, min(bb, h))
        return (l, t, rr, bb)

    l, t, rr, bb = clamp(shrunk)
    if rr <= l or bb <= t:
        # margin was too big for this cell, don't collapse it
        return clamp((left, top, right, bottom))
    return (l, t, rr, bb)


def crop_signature_box(image, rect):
    """Cut the signature cell out of the full sheet image.

    rect is (left, top, right, bottom). We reject rectangles that fall
    outside the image or have no area - that way a bad coordinate from
    upstream fails loudly here instead of silently returning an empty
    array."""
    _validate_image(image)
    left, top, right, bottom = [int(v) for v in rect]

    h, w = image.shape[:2]
    if left < 0 or top < 0 or right > w or bottom > h:
        raise ValueError(
            f"crop rectangle {rect} is outside image of size {(w, h)}")
    if right <= left or bottom <= top:
        raise ValueError(f"crop rectangle {rect} has no area")

    return image[top:bottom, left:right].copy()


def remove_table_borders(box_img, band=None, dark_value=128, dark_frac=0.5):
    """Whiten any leftover grid-line pixels around the edge of a crop.

    Even after the inner margin, a bit of the ruled border can sneak into
    the crop. A full dark row/column near the edge is almost certainly a
    table line (a real signature doesn't span the entire width), so we
    paint those white. `band` is how many pixels in from each edge we
    check - it scales with the box size if you don't pass one."""
    _validate_image(box_img)
    out = box_img.copy()

    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
    h, w = gray.shape

    if band is None:
        band = max(2, int(0.06 * min(h, w)))

    white = (255, 255, 255) if out.ndim == 3 else 255

    top_rows = range(0, min(band, h))
    bottom_rows = range(max(0, h - band), h)
    for row in list(top_rows) + list(bottom_rows):
        if np.mean(gray[row, :] < dark_value) > dark_frac:
            out[row, :] = white

    left_cols = range(0, min(band, w))
    right_cols = range(max(0, w - band), w)
    for col in list(left_cols) + list(right_cols):
        if np.mean(gray[:, col] < dark_value) > dark_frac:
            out[:, col] = white

    return out


def extract_all_signature_regions(image, table_result, source_name=None,
                                  margin=4, save=True, out_dir=OUTPUT_DIR):
    """The main entry point - turn one sheet + Member 4's result into a
    list of cleaned signature crops, one per student row and in order.

    Returns a list of dicts like:
        {"row_number": 1, "bbox": (l, t, r, b), "image": <ndarray>}

    Absent students still get an entry (their crop is just blank) so the
    row numbering stays aligned with info.xml downstream."""
    _validate_image(image)

    # Member 4's result comes in one of two shapes depending on whose
    # version we're wired to:
    #   - process_table_detection() -> {"rows", "signature_column", ...}
    #   - the boxes-based version    -> {"row_boundaries", "boxes"}
    # We accept either so this file drops in against both.
    rows_in = table_result.get("rows")
    if rows_in is None:
        rows_in = table_result.get("row_boundaries", [])
    rows = extract_student_rows(rows_in)

    sig_col = table_result.get("signature_column")
    if sig_col is None:
        sig_col = detect_signature_column(table_result.get("boxes", []))
    sig_left, sig_right = int(sig_col[0]), int(sig_col[1])

    blank = (np.full((1, 1, 3), 255, np.uint8)
             if image.ndim == 3 else np.full((1, 1), 255, np.uint8))

    regions = []
    for i, (top, bottom) in enumerate(rows, start=1):
        rect = add_inner_margin((sig_left, top, sig_right, bottom),
                                margin, image.shape)
        try:
            crop = crop_signature_box(image, rect)
            crop = remove_table_borders(crop)
        except ValueError:
            # keep the row so numbering doesn't drift; just hand back a blank
            crop = blank.copy()
        regions.append({"row_number": i, "bbox": rect, "image": crop})

    if save:
        os.makedirs(out_dir, exist_ok=True)
        prefix = source_name or "sheet"
        for r in regions:
            path = os.path.join(out_dir, f"{prefix}_row{r['row_number']}_signature.png")
            cv2.imwrite(path, r["image"])
        print(f"Saved {len(regions)} signature regions to {out_dir}")

    return regions


def make_test_scene(n_rows=4, signed_rows=(0, 2)):
    """Builds a fake sheet + a Member-4-style result so we can test this
    file on its own (and demo it without a real photo). Draws a table
    with a signature column on the right, scribbles a 'signature' in the
    rows listed in signed_rows and leaves the rest empty."""
    img = np.full((300, 400, 3), 255, np.uint8)

    tx, ty, tw, th = 20, 20, 360, 260
    sig_x, sig_w = 280, 100  # signature column
    cv2.rectangle(img, (tx, ty), (tx + tw, ty + th), (0, 0, 0), 2)

    row_boundaries = []
    boxes = [(tx, ty, tw, th)]
    row_h = 50
    for r in range(n_rows):
        top = ty + 20 + r * row_h
        bottom = top + row_h - 8
        row_boundaries.append((top, bottom))
        boxes.append((sig_x, top, sig_w, bottom - top))
        # draw the cell border
        cv2.rectangle(img, (sig_x, top), (sig_x + sig_w, bottom), (0, 0, 0), 1)
        if r in signed_rows:
            cv2.putText(img, "sig", (sig_x + 15, top + 30),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (200, 0, 0), 2)

    # return both shapes so the scene works whichever Member 4 API is used
    table_result = {
        "table_boundary": (tx, ty, tw, th),
        "rows": row_boundaries,
        "signature_column": (sig_x, sig_x + sig_w),
        "boxes": boxes,
        "row_boundaries": row_boundaries,
    }
    return img, table_result


def _run_on_real_sheet(name, corrected_dir=os.path.join("output", "corrected_images")):
    """End-to-end helper: load a corrected sheet, run the earlier members'
    steps to get a table result, then extract signature regions. Only
    used from the command line, not by the tests."""
    from image_preprocessing import preprocess_image
    from thresholding import otsu_threshold, morphological_cleaning
    import table_detection as td

    path = os.path.join(corrected_dir, f"{name}_corrected.png")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Corrected sheet not found: {path}")

    # run Members 2-4's steps to get the table layout for this sheet
    binary = morphological_cleaning(otsu_threshold(preprocess_image(img)))
    table_result = td.process_table_detection(binary, source_name=name)

    regions = extract_all_signature_regions(img, table_result, source_name=name)
    print(f"Sheet {name}: extracted {len(regions)} signature boxes")
    return regions


def run(sheet_names=None,
        corrected_dir=os.path.join("output", "corrected_images")):
    """Extract signature boxes for a list of sheets. If sheet_names is
    None we process every corrected sheet we can find. Bad/missing sheets
    are skipped with a message instead of crashing the whole run."""
    if not sheet_names:
        if os.path.isdir(corrected_dir):
            sheet_names = sorted(
                f[:-len("_corrected.png")]
                for f in os.listdir(corrected_dir)
                if f.endswith("_corrected.png")
            )
        else:
            sheet_names = []

    if not sheet_names:
        print(f"No corrected sheets found in {corrected_dir}")
        return {}

    results = {}
    for name in sheet_names:
        try:
            results[name] = _run_on_real_sheet(name, corrected_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping sheet {name}: {e}")
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # real sheets, e.g:  python signature_extraction.py 1 2 3
        run(sys.argv[1:])
    else:
        # no real image given - use the synthetic scene so we can still
        # see something get produced without waiting on real photos
        print("no sheet given, using a synthetic test scene instead")
        img, table_result = make_test_scene()
        regions = extract_all_signature_regions(img, table_result, source_name="demo")
        for r in regions:
            print(f"  row {r['row_number']}: bbox={r['bbox']} "
                  f"size={r['image'].shape[:2]}")
