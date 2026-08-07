#!/usr/bin/env python3
"""
Visual signature comparison across all sheets for a given student.
Usage: python scripts/compare_signatures.py <student_index>
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.signature_verification import (
    normalize_signature,
    resize_signature,
    calculate_ssim_score,
    calculate_orb_score,
)


def get_row_for_student(student_index, xml_path):
    """Look up row number for a student from info.xml."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    students = root.find("students")
    for i, student in enumerate(students.findall("student"), start=1):
        idx = student.find("index").text
        if str(idx) == str(student_index):
            return i
    return None


def main():
    parser = argparse.ArgumentParser(description="Visual signature comparison for a student.")
    parser.add_argument("student_index", help="Student index, e.g. 001 or 10000409")
    args = parser.parse_args()

    student = args.student_index

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    ref_base = os.path.join(project_root, "data", "reference_signatures", student)
    regions_dir = os.path.join(project_root, "output", "signature_regions")
    output_dir = os.path.join(project_root, "output", "signature_reports")
    xml_path = os.path.join(project_root, "data", "info.xml")

    os.makedirs(output_dir, exist_ok=True)

    # Get row number from XML
    row_number = get_row_for_student(student, xml_path)
    if row_number is None:
        print(f"Student {student} not found in {xml_path}")
        sys.exit(1)

    # Find reference image (first available ref_*.png)
    ref_files = sorted([f for f in os.listdir(ref_base) if f.startswith("ref_")]) if os.path.isdir(ref_base) else []
    if not ref_files:
        print(f"No reference signatures found for student {student}")
        sys.exit(1)

    ref_path = os.path.join(ref_base, ref_files[0])
    ref = cv2.imread(ref_path)
    if ref is None:
        print(f"Could not load reference: {ref_path}")
        sys.exit(1)

    sheets = ["sheet1_students", "sheet2_students", "sheet3_students", "sheet4_students", "sheet5_students"]

    fig, axes = plt.subplots(1, len(sheets) + 1, figsize=(18, 3))

    # Show reference
    axes[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Reference\n({ref_files[0]})")
    axes[0].axis("off")

    for i, sheet in enumerate(sheets):
        candidate_path = os.path.join(regions_dir, sheet, f"row_{row_number:02d}.png")
        candidate = cv2.imread(candidate_path)

        if candidate is not None:
            axes[i + 1].imshow(cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB))

            ssim_val = calculate_ssim_score(ref, candidate)
            orb_val = calculate_orb_score(ref, candidate)
            combined = ssim_val * 0.6 + orb_val * 0.4

            axes[i + 1].set_title(
                f"{sheet}\nSSIM:{ssim_val:.2f}\nORB:{orb_val:.2f}\nCombined:{combined:.2f}"
            )
            axes[i + 1].axis("off")
        else:
            axes[i + 1].set_title(f"{sheet}\nNot found")
            axes[i + 1].axis("off")

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"signature_comparison_{student}.png")
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()