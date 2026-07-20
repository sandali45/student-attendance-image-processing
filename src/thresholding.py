# src/thresholding.py
# Member 3 - Binarization
#
# This file takes the grayscale image from Member 2 and turns it into
# a black and white (binary) image. We try 3 different thresholding
# methods, clean up the result a bit, and then pick whichever one
# looks best.
#
# Rule I'm using everywhere: after thresholding, the pixel is either
# 0 (black = ink / signature / table lines) or 255 (white = paper).

import os
import cv2
import numpy as np


def check_image(img):
    """Small helper to make sure we got a proper grayscale image.
    Not super fancy, just enough so the program doesn't crash if
    something empty/None gets passed in from another member's code."""
    if img is None:
        raise ValueError("image is None")
    if img.size == 0:
        raise ValueError("image is empty")
    if len(img.shape) == 3:
        # in case someone passes a color image by mistake
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def global_threshold(gray_img, thresh_value=127):
    """Basic thresholding - pixels above thresh_value become white,
    everything else becomes black. Easy but doesn't handle shadows
    well."""
    gray_img = check_image(gray_img)
    ret, binary_img = cv2.threshold(gray_img, thresh_value, 255, cv2.THRESH_BINARY)
    return binary_img


def otsu_threshold(gray_img):
    """Otsu's method picks the threshold value automatically based on
    the histogram of the image. Usually better than just guessing a
    number like we do in global_threshold()."""
    gray_img = check_image(gray_img)
    # the 0 here doesn't actually matter, OTSU flag overrides it
    ret, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_img


def adaptive_threshold(gray_img, block_size=25, c=10):
    """Instead of one threshold for the whole image, this works out a
    threshold for each small region. Should help when the photo has
    shadows or uneven lighting across the sheet."""
    gray_img = check_image(gray_img)

    # block_size has to be odd or cv2 throws an error
    if block_size % 2 == 0:
        block_size += 1

    binary_img = cv2.adaptiveThreshold(
        gray_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )
    return binary_img


def morphological_cleaning(binary_img, kernel_size=3):
    """Gets rid of tiny bits of noise (single stray black pixels etc)
    using opening + closing. Kept it simple with one function that
    does both instead of splitting into two."""
    binary_img = check_image(binary_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # ink is black (0), but morphology functions in cv2 assume the
    # "object" is white, so we invert first and invert back after
    inverted = cv2.bitwise_not(binary_img)
    opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.bitwise_not(closed)

    # make sure it's still strictly 0/255 after all that
    ret, cleaned = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
    return cleaned


def ink_ratio(binary_img):
    """how much of the image is ink (black pixels) vs paper"""
    return np.sum(binary_img == 0) / binary_img.size


def count_noise_blobs(binary_img, max_size=15):
    """counts how many really small connected blobs of ink there are.
    Lots of tiny blobs usually = noise, not real signatures."""
    inverted = cv2.bitwise_not(binary_img)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    small_blobs = 0
    for i in range(1, n_labels):  # label 0 is background, skip it
        if stats[i, cv2.CC_STAT_AREA] <= max_size:
            small_blobs += 1
    return small_blobs


def select_best_binary_image(candidates):
    """Looks at all the candidate binary images and picks the "best"
    one. My scoring is pretty basic:
      - if the image is mostly black or mostly white, that's probably
        wrong (threshold wiped out the signature, or turned the whole
        page black), so we penalize that
      - if there's a lot of tiny noise blobs, penalize that too

    Lower score = better. candidates should be a dict like:
    {"global": img1, "otsu": img2, "adaptive": img3}
    """
    if len(candidates) == 0:
        raise ValueError("no candidates given")

    scores = {}
    for name in candidates:
        img = candidates[name]
        r = ink_ratio(img)

        # a normal signing sheet is mostly white paper with a bit of
        # ink, so ideal ratio is roughly between 1% and 35%
        if r < 0.01:
            penalty = (0.01 - r) * 10
        elif r > 0.35:
            penalty = (r - 0.35) * 10
        else:
            penalty = 0

        noise = count_noise_blobs(img)
        score = penalty + noise * 0.01
        scores[name] = score

    # find the name with the lowest score
    best_name = min(scores, key=scores.get)
    return best_name, candidates[best_name], scores


def make_test_image():
    """Just makes a fake grayscale image to test with, since Member 2
    hasn't given us real output yet. Has a lighting gradient (like a
    shadow on one side) and some fake handwriting."""
    h, w = 300, 500
    gradient = np.tile(np.linspace(220, 110, w), (h, 1)).astype(np.uint8)
    img = gradient.copy()

    cv2.rectangle(img, (20, 20), (w - 20, h - 20), 40, 2)
    for y in [90, 160, 230]:
        cv2.line(img, (20, y), (w - 20, y), 40, 1)

    cv2.putText(img, "JSnow", (60, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 30, 2)
    cv2.putText(img, "JamesBond", (60, 140), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 30, 2)
    # row 3 left empty on purpose - that student was absent

    return img


def run(gray_img, out_dir="output/binary_images", source_name=None):
    """Runs all the steps and saves the images. This is basically what
    the whole file is for."""
    gray_img = check_image(gray_img)
    os.makedirs(out_dir, exist_ok=True)

    prefix = "demo"
    if source_name is not None:
        prefix = os.path.splitext(os.path.basename(source_name))[0]

    g = global_threshold(gray_img)
    o = otsu_threshold(gray_img)
    a = adaptive_threshold(gray_img)

    candidates = {"global": g, "otsu": o, "adaptive": a}
    best_name, best_img, scores = select_best_binary_image(candidates)
    cleaned = morphological_cleaning(best_img)

    cv2.imwrite(os.path.join(out_dir, f"{prefix}_global_threshold.png"), g)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_otsu_threshold.png"), o)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_adaptive_threshold.png"), a)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_cleaned_binary.png"), cleaned)

    print("scores:", scores)
    print("best method:", best_name)
    print("saved images to", out_dir)


if __name__ == "__main__":
    # if we're given a real image path, use that, otherwise just make
    # a fake one so we can test on our own without waiting for
    # Member 2 to finish
    import sys

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        run(img, source_name=img_path)
    else:
        img = make_test_image()
        os.makedirs("output/grayscale_images", exist_ok=True)
        cv2.imwrite("output/grayscale_images/demo_grayscale.png", img)
        print("no image given, using a fake test image instead")
        run(img)