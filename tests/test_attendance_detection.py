import cv2
import numpy as np
import pytest

from src.attendance_detection import (
    calculate_ink_ratio,
    count_contours,
    remove_small_noise,
    classify_present_absent,
    classify_attendance,
)


def create_blank_box():
    """Create a completely white signature box."""

    return np.full(
        (100, 300, 3),
        255,
        dtype=np.uint8,
    )


def create_signed_box():
    """Create a synthetic black signature."""

    image = create_blank_box()

    # Create several connected signature-like strokes.
    cv2.line(
        image,
        (50, 50),
        (240, 35),
        (0, 0, 0),
        5,
    )

    cv2.line(
        image,
        (70, 65),
        (220, 30),
        (0, 0, 0),
        5,
    )

    cv2.ellipse(
        image,
        (145, 50),
        (65, 25),
        0,
        0,
        300,
        (0, 0, 0),
        5,
    )

    return image


def create_border_only_box():
    """Create a box containing only table borders."""

    image = create_blank_box()

    cv2.rectangle(
        image,
        (0, 0),
        (299, 99),
        (0, 0, 0),
        3,
    )

    return image


def create_noise_only_box():
    """Create several tiny black dots."""

    image = create_blank_box()

    points = [
        (80, 30),
        (110, 60),
        (150, 40),
        (190, 70),
        (230, 35),
    ]

    for point in points:
        cv2.circle(
            image,
            point,
            1,
            (0, 0, 0),
            -1,
        )

    return image


def create_light_blue_signature():
    """Create a lighter coloured pen signature."""

    image = create_blank_box()

    blue = (220, 120, 40)

    cv2.line(
        image,
        (50, 45),
        (250, 35),
        blue,
        5,
    )

    cv2.line(
        image,
        (60, 65),
        (230, 25),
        blue,
        5,
    )

    cv2.ellipse(
        image,
        (150, 50),
        (60, 25),
        0,
        0,
        300,
        blue,
        5,
    )

    return image


def test_blank_box_is_absent():

    image = create_blank_box()

    result = classify_present_absent(
        image,
        row_number=1,
    )

    assert result["present"] is False
    assert result["status"] == "Absent"


def test_signed_box_is_present():

    image = create_signed_box()

    result = classify_present_absent(
        image,
        row_number=1,
        threshold=0.01,
    )

    assert result["present"] is True
    assert result["status"] == "Present"
    assert result["score"] > 0


def test_border_only_is_absent():

    image = create_border_only_box()

    result = classify_present_absent(
        image,
        row_number=1,
    )

    assert result["present"] is False


def test_small_noise_is_absent():

    image = create_noise_only_box()

    result = classify_present_absent(
        image,
        row_number=1,
    )

    assert result["present"] is False


def test_light_coloured_signature_can_be_detected():

    image = create_light_blue_signature()

    result = classify_present_absent(
        image,
        row_number=1,
        threshold=0.01,
    )

    assert result["present"] is True


def test_invalid_threshold_zero():

    image = create_signed_box()

    with pytest.raises(ValueError):
        classify_present_absent(
            image,
            row_number=1,
            threshold=0,
        )


def test_invalid_threshold_above_one():

    image = create_signed_box()

    with pytest.raises(ValueError):
        classify_present_absent(
            image,
            row_number=1,
            threshold=1.5,
        )


def test_invalid_row_number():

    image = create_signed_box()

    with pytest.raises(ValueError):
        classify_present_absent(
            image,
            row_number=0,
        )


def test_none_image_is_rejected():

    with pytest.raises(ValueError):
        classify_present_absent(
            None,
            row_number=1,
        )


def test_calculate_ink_ratio_blank():

    image = create_blank_box()

    ratio = calculate_ink_ratio(image)

    assert ratio == 0


def test_signed_box_has_ink():

    image = create_signed_box()

    ratio = calculate_ink_ratio(image)

    assert ratio > 0


def test_signed_box_has_contours():

    image = create_signed_box()

    number_of_contours = count_contours(image)

    assert number_of_contours > 0


def test_remove_small_noise():

    mask = np.zeros(
        (100, 300),
        dtype=np.uint8,
    )

    # Tiny noise
    mask[40, 40] = 255
    mask[50, 80] = 255

    cleaned = remove_small_noise(
        mask,
        min_area=8,
    )

    assert cv2.countNonZero(cleaned) == 0


def test_multiple_signature_regions():

    regions = [
        {
            "row_number": 1,
            "image": create_signed_box(),
        },
        {
            "row_number": 2,
            "image": create_blank_box(),
        },
    ]

    results = classify_attendance(
        regions,
        threshold=0.01,
    )

    assert len(results) == 2

    assert results[0]["row_number"] == 1
    assert results[0]["present"] is True

    assert results[1]["row_number"] == 2
    assert results[1]["present"] is False