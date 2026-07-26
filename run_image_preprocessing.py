from pathlib import Path
import cv2

from src.image_preprocessing import (
    convert_to_grayscale,
    apply_gaussian_filter,
    apply_median_filter,
    enhance_contrast,
    save_image,
)

# -------------------------------------------------------
# Input and Output Directories
# -------------------------------------------------------

INPUT_DIR = Path(
    "output/corrected_images"
)

OUTPUT_DIR = Path(
    "output/grayscale_images"
)

# Create output folders
GRAY_DIR = OUTPUT_DIR / "grayscale"
GAUSSIAN_DIR = OUTPUT_DIR / "gaussian"
MEDIAN_DIR = OUTPUT_DIR / "median"
ENHANCED_DIR = OUTPUT_DIR / "enhanced"

for directory in [
    GRAY_DIR,
    GAUSSIAN_DIR,
    MEDIAN_DIR,
    ENHANCED_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Supported image formats
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

print("Starting image preprocessing...\n")

for image_path in INPUT_DIR.iterdir():

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Skipping unreadable image: {image_path.name}")
        continue

    # -------------------------
    # Stage 1 - Grayscale
    # -------------------------
    gray = convert_to_grayscale(image)
    save_image(gray, GRAY_DIR / image_path.name)

    # -------------------------
    # Stage 2 - Gaussian Blur
    # -------------------------
    gaussian = apply_gaussian_filter(gray)
    save_image(gaussian, GAUSSIAN_DIR / image_path.name)

    # -------------------------
    # Stage 3 - Median Filter
    # -------------------------
    median = apply_median_filter(gaussian)
    save_image(median, MEDIAN_DIR / image_path.name)

    # -------------------------
    # Stage 4 - Contrast Enhancement
    # -------------------------
    enhanced = enhance_contrast(median)
    save_image(enhanced, ENHANCED_DIR / image_path.name)

    print(f"Processed: {image_path.name}")

print("\nAll images processed successfully!")