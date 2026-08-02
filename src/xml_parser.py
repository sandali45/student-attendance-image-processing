"""
src/xml_parser.py
Member 7 — XML file reading and student-row mapping.
"""

import xml.etree.ElementTree as ET
import os


def read_xml_file(filepath):
    """Parse the XML file and return the root Element. Raises clear errors on failure."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"XML file not found: {filepath}")

    try:
        tree = ET.parse(filepath)
    except ET.ParseError as e:
        raise ValueError(f"Invalid/broken XML file: {filepath} ({e})")

    return tree.getroot()


def extract_subject_information(root):
    """Return a dict with subject_name and subject_code from the XML root."""
    subject_elem = root.find("subject")
    if subject_elem is None:
        raise ValueError("No <subject> element found in XML")

    name_elem = subject_elem.find("subject_name")
    code_elem = subject_elem.find("subject_code")

    return {
        "subject_name": name_elem.text.strip() if name_elem is not None and name_elem.text else None,
        "subject_code": code_elem.text.strip() if code_elem is not None and code_elem.text else None,
    }


def extract_student_records(root):
    """Return a list of {"student_index": str, "student_name": str} in document order."""
    students_elem = root.find("students")
    if students_elem is None:
        raise ValueError("No <students> element found in XML")

    records = []
    for student in students_elem.findall("student"):
        index_elem = student.find("index")
        name_elem = student.find("name")

        student_index = index_elem.text.strip() if index_elem is not None and index_elem.text else None
        student_name = name_elem.text.strip() if name_elem is not None and name_elem.text else None

        records.append({
            "student_index": student_index,
            "student_name": student_name,
        })

    return records


def validate_student_records(records):
    """
    Check for duplicate indices and missing names/indices.
    Returns a list of error message strings (empty list = no problems).
    """
    errors = []
    seen_indices = set()

    for i, record in enumerate(records):
        idx = record.get("student_index")
        name = record.get("student_name")

        if not idx:
            errors.append(f"Record {i}: missing student_index")
        elif idx in seen_indices:
            errors.append(f"Duplicate student_index found: {idx}")
        else:
            seen_indices.add(idx)

        if not name:
            errors.append(f"Record {i}: missing student_name (index={idx})")

    return errors


def map_rows_to_students(records, row_count=None):
    """
    Map each student record to its table row number (1-based, in document order).
    If row_count is given, checks it matches len(records) and raises if not.

    Returns a list of dicts: {"row_number": int, "student_index": str, "student_name": str}
    """
    if row_count is not None and row_count != len(records):
        raise ValueError(
            f"Row count mismatch: detected {row_count} rows but XML has {len(records)} students"
        )

    mapped = []
    for i, record in enumerate(records, start=1):
        mapped.append({
            "row_number": i,
            "student_index": record["student_index"],
            "student_name": record["student_name"],
        })

    return mapped


if __name__ == "__main__":
    # Quick manual smoke test, no dependency on other members.
    root = read_xml_file("data/info.xml")
    subject_info = extract_subject_information(root)
    records = extract_student_records(root)
    errors = validate_student_records(records)
    mapped = map_rows_to_students(records)

    print("Subject:", subject_info)
    print("Errors:", errors)
    print("Mapped rows:", mapped)