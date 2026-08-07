from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from src.image_loader import process_sheet_image
from src.image_preprocessing import (
    convert_to_grayscale,
    apply_gaussian_filter,
    apply_median_filter,
    enhance_contrast,
)
from src.thresholding import (
    global_threshold,
    otsu_threshold,
    adaptive_threshold,
    select_best_binary_image,
    morphological_cleaning,
)
from src.table_detection import process_table_detection
from src.signature_extraction import extract_all_signature_regions
from src.attendance_detection import classify_attendance
from src.xml_parser import (
    read_xml_file,
    extract_subject_information,
    extract_student_records,
    validate_student_records,
    map_rows_to_students,
)
from src.database import (
    create_database,
    create_tables,
    insert_student,
    save_attendance_record,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output"
DEFAULT_DB_PATH = ROOT / "database" / "attendance.db"


def _progress(callback, message: str) -> None:
    print(message)
    if callback is not None:
        callback(message)


def _save_image(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Could not save image: {path}")
    return str(path)


def _select_student_rows(rows: list[tuple[int, int]], student_count: int) -> list[tuple[int, int]]:
    """Select exactly the student rows from Member 4's raw row detections.

    The supplied sheets can include the lecturer table/header and some signatures
    create false horizontal lines. Student rows are the lower block of similarly
    sized rows. Tiny split rows are merged until exactly student_count rows remain.
    """
    if student_count < 1:
        raise ValueError("XML must contain at least one student.")
    if len(rows) < student_count:
        raise ValueError(
            f"Detected only {len(rows)} row regions, but XML contains {student_count} students."
        )

    rows = sorted((int(t), int(b)) for t, b in rows if int(b) > int(t))
    heights = np.array([b - t for t, b in rows], dtype=float)
    median_h = float(np.median(heights))
    normal_heights = heights[heights >= max(4.0, median_h * 0.75)]
    typical_h = float(np.median(normal_heights)) if normal_heights.size else median_h

    bottom = rows[-1][1]
    target_top = bottom - (student_count * typical_h)
    start_idx = min(range(len(rows)), key=lambda i: abs(rows[i][0] - target_top))
    selected = rows[start_idx:]

    # If we still have too many segments, merge the adjacent pair whose combined
    # height is closest to a normal student-row height. This repairs rows split by
    # a false horizontal line from handwriting.
    while len(selected) > student_count:
        best_i = None
        best_cost = None
        for i in range(len(selected) - 1):
            combined_h = selected[i + 1][1] - selected[i][0]
            gap = selected[i + 1][0] - selected[i][1]
            cost = abs(combined_h - typical_h) + abs(gap) * 3
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_i = i
        i = best_i
        selected[i:i + 2] = [(selected[i][0], selected[i + 1][1])]

    # If nearest-top selection produced too few, move the start upward.
    while len(selected) < student_count and start_idx > 0:
        start_idx -= 1
        selected.insert(0, rows[start_idx])

    if len(selected) != student_count:
        raise ValueError(
            f"Could not isolate exactly {student_count} student rows from {len(rows)} raw rows."
        )

    return selected


def _get_or_create_session(conn, subject_name: str, session_date: str, sheet_filename: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT subject_id FROM subjects WHERE subject_name = ?", (subject_name,))
    row = cur.fetchone()
    if row:
        subject_id = row[0]
    else:
        cur.execute("INSERT INTO subjects (subject_name) VALUES (?)", (subject_name,))
        subject_id = cur.lastrowid

    cur.execute(
        """
        SELECT session_id FROM attendance_sessions
        WHERE subject_id = ? AND date = ? AND sheet_filename = ?
        ORDER BY session_id DESC LIMIT 1
        """,
        (subject_id, session_date, sheet_filename),
    )
    row = cur.fetchone()
    if row:
        conn.commit()
        return int(row[0])

    cur.execute(
        """
        INSERT INTO attendance_sessions (subject_id, date, sheet_filename)
        VALUES (?, ?, ?)
        """,
        (subject_id, session_date, sheet_filename),
    )
    conn.commit()
    return int(cur.lastrowid)


def process_attendance(
    image_path: str,
    xml_path: str,
    session_date: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    threshold: float = 0.020,
    save_to_db: bool = True,
    progress_callback=None,
) -> dict:
    image_path = str(Path(image_path))
    xml_path = str(Path(xml_path))
    session_date = session_date or date.today().isoformat()
    stem = Path(image_path).stem

    _progress(progress_callback, "1/9 Reading XML student information...")
    root = read_xml_file(xml_path)
    subject = extract_subject_information(root)
    students = extract_student_records(root)
    errors = validate_student_records(students)
    if errors:
        raise ValueError("XML validation failed: " + "; ".join(errors))

    _progress(progress_callback, "2/9 Loading and correcting signing-sheet photo...")
    corrected = process_sheet_image(image_path)
    corrected_path = OUTPUT_ROOT / "corrected_images" / f"{stem}_corrected.png"

    _progress(progress_callback, "3/9 Grayscale, filtering and contrast enhancement...")
    gray = convert_to_grayscale(corrected)
    gaussian = apply_gaussian_filter(gray)
    median = apply_median_filter(gaussian)
    enhanced = enhance_contrast(median)

    gray_path = OUTPUT_ROOT / "grayscale_images" / "grayscale" / f"{stem}.png"
    gaussian_path = OUTPUT_ROOT / "grayscale_images" / "gaussian" / f"{stem}.png"
    median_path = OUTPUT_ROOT / "grayscale_images" / "median" / f"{stem}.png"
    enhanced_path = OUTPUT_ROOT / "grayscale_images" / "enhanced" / f"{stem}.png"
    _save_image(gray_path, gray)
    _save_image(gaussian_path, gaussian)
    _save_image(median_path, median)
    _save_image(enhanced_path, enhanced)

    _progress(progress_callback, "4/9 Binarizing image...")
    g = global_threshold(enhanced)
    o = otsu_threshold(enhanced)
    a = adaptive_threshold(enhanced)
    best_name, best_binary, _scores = select_best_binary_image({
        "global": g,
        "otsu": o,
        "adaptive": a,
    })
    cleaned = morphological_cleaning(best_binary)

    binary_dir = OUTPUT_ROOT / "binary_images"
    _save_image(binary_dir / f"{stem}_global_threshold.png", g)
    _save_image(binary_dir / f"{stem}_otsu_threshold.png", o)
    _save_image(binary_dir / f"{stem}_adaptive_threshold.png", a)
    cleaned_path = binary_dir / f"{stem}_cleaned_binary.png"
    _save_image(cleaned_path, cleaned)

    _progress(progress_callback, "5/9 Detecting attendance table and student rows...")
    table_result = process_table_detection(cleaned, source_name=stem)
    raw_row_count = len(table_result["rows"])
    table_result["rows"] = _select_student_rows(table_result["rows"], len(students))

    _progress(progress_callback, f"6/9 Extracting {len(students)} student signature regions...")
    regions = extract_all_signature_regions(
        corrected,
        table_result,
        source_name=stem,
        save=True,
        out_dir=str(OUTPUT_ROOT / "signature_regions"),
    )

    _progress(progress_callback, "7/9 Detecting Present / Absent...")
    attendance = classify_attendance(regions, threshold=threshold)
    mapped = map_rows_to_students(students, row_count=len(attendance))

    combined = []
    for student, detection in zip(mapped, attendance):
        combined.append({
            "row_number": student["row_number"],
            "student_index": student["student_index"],
            "student_name": student["student_name"],
            "present": detection["present"],
            "status": detection["status"],
            "score": detection["score"],
            "contour_count": detection["contour_count"],
            "threshold": detection["threshold"],
        })

    if save_to_db:
        _progress(progress_callback, "8/9 Saving attendance to SQLite database...")
        conn = create_database(str(db_path))
        try:
            create_tables(conn)
            for student in students:
                insert_student(conn, student["student_index"], student["student_name"])
            session_id = _get_or_create_session(
                conn,
                subject.get("subject_name") or "Unknown Subject",
                session_date,
                Path(image_path).name,
            )
            for row in combined:
                save_attendance_record(
                    conn,
                    session_id,
                    row["student_index"],
                    row["present"],
                    row["score"],
                )
        finally:
            conn.close()
    else:
        _progress(progress_callback, "8/9 Database saving skipped.")

    _progress(progress_callback, "9/9 Saving final attendance result...")
    result = {
        "subject_name": subject.get("subject_name"),
        "subject_code": subject.get("subject_code"),
        "date": session_date,
        "sheet": Path(image_path).name,
        "raw_rows_detected": raw_row_count,
        "student_rows_used": len(regions),
        "threshold_method": best_name,
        "students": combined,
        "stages": {
            "Original": image_path,
            "Corrected": str(corrected_path),
            "Grayscale": str(gray_path),
            "Gaussian": str(gaussian_path),
            "Median": str(median_path),
            "Enhanced": str(enhanced_path),
            "Binary": str(cleaned_path),
        },
    }

    result_path = OUTPUT_ROOT / "attendance_results" / f"{stem}_final_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["result_path"] = str(result_path)
    _progress(progress_callback, "Completed successfully.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Student Attendance Management System")
    parser.add_argument("image", help="Signing sheet image path")
    parser.add_argument("xml", help="Student/subject XML file path")
    parser.add_argument("--date", dest="session_date", default=None, help="Session date YYYY-MM-DD")
    parser.add_argument("--threshold", type=float, default=0.020)
    parser.add_argument("--no-db", action="store_true", help="Do not save to database")
    args = parser.parse_args()

    try:
        result = process_attendance(
            args.image,
            args.xml,
            session_date=args.session_date,
            threshold=args.threshold,
            save_to_db=not args.no_db,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    print("\nAttendance Summary")
    print("-" * 80)
    for row in result["students"]:
        print(
            f"{row['row_number']:02d} | {row['student_index']:<10} | "
            f"{row['student_name']:<35} | {row['status']:<7} | {row['score']:.4f}"
        )
    print("-" * 80)
    print(f"Saved: {result['result_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
