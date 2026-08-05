#!/usr/bin/env python3
"""
investigate.py
Member 10 -- Signature verification entry point.

Usage:
    python investigate.py 10000409
"""

import argparse
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.signature_verification import compare_with_reference_signatures
from src.xml_parser import (
    read_xml_file,
    extract_student_records,
    map_rows_to_students,
)

SIGNATURE_REGIONS_DIR = "output/signature_regions"
REFERENCE_SIGNATURES_DIR = "data/reference_signatures"
REPORT_OUTPUT_DIR = "output/signature_reports"
INFO_XML_PATH = "data/info.xml"
MATCH_THRESHOLD = 0.65

ROW_FILENAME_PATTERN = re.compile(r"row_(\d+)\.png$", re.IGNORECASE)


def get_row_for_student(student_index, xml_path=INFO_XML_PATH):
    root = read_xml_file(xml_path)
    records = extract_student_records(root)
    mapped = map_rows_to_students(records)

    for entry in mapped:
        if str(entry["student_index"]) == str(student_index):
            return entry["row_number"]
    return None


def find_candidate_signatures(row_number, region_dir=SIGNATURE_REGIONS_DIR):
    if row_number is None or not os.path.isdir(region_dir):
        return []

    found = []

    for dirpath, _, filenames in os.walk(region_dir):
        for fname in filenames:
            match = ROW_FILENAME_PATTERN.search(fname)

            if match and int(match.group(1)) == row_number:
                sheet_label = os.path.basename(dirpath)

                found.append(
                    (sheet_label, os.path.join(dirpath, fname))
                )

    found.sort(key=lambda pair: pair[0])

    # Use only the latest sheet as the candidate
    if found:
        return [found[-1]]

    return []


def write_report(path, student_index, row_number, per_sheet_results, overall_score, overall_decision):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"Student: {student_index}\n")
        f.write(f"Row number: {row_number}\n")
        f.write(f"Sheets checked: {len(per_sheet_results)}\n\n")

        for sheet_label, candidate_path, result in per_sheet_results:
            f.write(f"[{sheet_label}] {candidate_path}\n")
            f.write(f"  References compared: {result['reference_count']}\n")
            for r in result["results"]:
                f.write(
                    f"    - {r['reference_file']}: "
                    f"SSIM={r['ssim_score']}, ORB={r['orb_score']}, "
                    f"Combined={r['combined_score']}\n"
                )
            f.write(f"  Best match: {result['best_reference']}\n")
            f.write(f"  Score: {result['best_score']}\n")
            f.write(f"  Decision: {result['decision']}\n\n")

        f.write(f"Overall similarity score: {overall_score}\n")
        f.write(f"Overall decision: {overall_decision}\n")


def main():
    parser = argparse.ArgumentParser(description="Verify a student's signature against references.")
    parser.add_argument("student_index", help="Student index, e.g. 10000409")
    args = parser.parse_args()

    student_index = args.student_index
    row_number = get_row_for_student(student_index)

    print(f"Student: {student_index}")

    if row_number is None:
        print("Similarity score: N/A")
        print("Decision: Student Not Found In XML")
        return

    candidates = find_candidate_signatures(row_number)

    if not candidates:
        print("Similarity score: N/A")
        print("Decision: No Extracted Signature Found")
        return

    reference_dir = os.path.join(REFERENCE_SIGNATURES_DIR, student_index)

    per_sheet_results = []
    for sheet_label, candidate_path in candidates:
        result = compare_with_reference_signatures(candidate_path, reference_dir, MATCH_THRESHOLD)
        per_sheet_results.append((sheet_label, candidate_path, result))
        print(f"  [{sheet_label}] score={result['best_score']:.2f} -> {result['decision']}")

    scored = [r for _, _, r in per_sheet_results if r["reference_count"] > 0]

    if not scored:
        overall_score = 0.0
        overall_decision = per_sheet_results[0][2]["decision"]
    else:
        overall_score = round(sum(r["best_score"] for r in scored) / len(scored), 4)
        overall_decision = "Matching" if overall_score >= MATCH_THRESHOLD else "Not Matching"

    print(f"Similarity score: {overall_score:.2f}")
    print(f"Decision: {overall_decision}")

    report_path = os.path.join(REPORT_OUTPUT_DIR, f"{student_index}_report.txt")
    write_report(report_path, student_index, row_number, per_sheet_results, overall_score, overall_decision)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()