"""
tests/test_xml_parser.py
Tests for Member 7 — xml_parser.py
"""

import pytest
from src.xml_parser import (
    read_xml_file,
    extract_subject_information,
    extract_student_records,
    validate_student_records,
    map_rows_to_students,
)


VALID_XML = """<?xml version="1.0" encoding="UTF-8"?>
<attendance_info>
    <subject>
        <subject_name>Math</subject_name>
        <subject_code>MTH101</subject_code>
    </subject>
    <students>
        <student><index>001</index><name>John Snow</name></student>
        <student><index>007</index><name>James Bond</name></student>
        <student><index>009</index><name>Andare</name></student>
    </students>
</attendance_info>"""

DUPLICATE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<attendance_info>
    <subject><subject_name>Math</subject_name><subject_code>MTH101</subject_code></subject>
    <students>
        <student><index>001</index><name>John Snow</name></student>
        <student><index>001</index><name>Duplicate Guy</name></student>
    </students>
</attendance_info>"""

MISSING_FIELD_XML = """<?xml version="1.0" encoding="UTF-8"?>
<attendance_info>
    <subject><subject_name>Math</subject_name><subject_code>MTH101</subject_code></subject>
    <students>
        <student><index></index><name>No Index Guy</name></student>
    </students>
</attendance_info>"""

BROKEN_XML = "<attendance_info><students><student>"


@pytest.fixture
def valid_xml_file(tmp_path):
    f = tmp_path / "info.xml"
    f.write_text(VALID_XML)
    return str(f)


@pytest.fixture
def duplicate_xml_file(tmp_path):
    f = tmp_path / "dup.xml"
    f.write_text(DUPLICATE_XML)
    return str(f)


@pytest.fixture
def missing_field_xml_file(tmp_path):
    f = tmp_path / "missing.xml"
    f.write_text(MISSING_FIELD_XML)
    return str(f)


@pytest.fixture
def broken_xml_file(tmp_path):
    f = tmp_path / "broken.xml"
    f.write_text(BROKEN_XML)
    return str(f)


def test_read_valid_xml(valid_xml_file):
    root = read_xml_file(valid_xml_file)
    assert root.tag == "attendance_info"


def test_read_missing_file():
    with pytest.raises(FileNotFoundError):
        read_xml_file("does/not/exist.xml")


def test_read_broken_xml(broken_xml_file):
    with pytest.raises(ValueError):
        read_xml_file(broken_xml_file)


def test_extract_subject_information(valid_xml_file):
    root = read_xml_file(valid_xml_file)
    info = extract_subject_information(root)
    assert info["subject_name"] == "Math"
    assert info["subject_code"] == "MTH101"


def test_extract_student_records_order(valid_xml_file):
    root = read_xml_file(valid_xml_file)
    records = extract_student_records(root)
    assert len(records) == 3
    assert records[0]["student_index"] == "001"
    assert records[1]["student_index"] == "007"
    assert records[2]["student_index"] == "009"


def test_validate_no_errors(valid_xml_file):
    root = read_xml_file(valid_xml_file)
    records = extract_student_records(root)
    errors = validate_student_records(records)
    assert errors == []


def test_validate_duplicate_index(duplicate_xml_file):
    root = read_xml_file(duplicate_xml_file)
    records = extract_student_records(root)
    errors = validate_student_records(records)
    assert any("Duplicate" in e for e in errors)


def test_validate_missing_index(missing_field_xml_file):
    root = read_xml_file(missing_field_xml_file)
    records = extract_student_records(root)
    errors = validate_student_records(records)
    assert any("missing student_index" in e for e in errors)


def test_map_rows_to_students(valid_xml_file):
    root = read_xml_file(valid_xml_file)
    records = extract_student_records(root)
    mapped = map_rows_to_students(records)
    assert mapped[0] == {"row_number": 1, "student_index": "001", "student_name": "John Snow"}
    assert mapped[2]["student_name"] == "Andare"


def test_map_rows_row_count_mismatch(valid_xml_file):
    root = read_xml_file(valid_xml_file)
    records = extract_student_records(root)
    with pytest.raises(ValueError):
        map_rows_to_students(records, row_count=5)