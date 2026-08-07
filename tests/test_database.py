"""
tests/test_database.py
Tests for src/database.py (Member 8).

Run with: pytest tests/test_database.py
"""

import os
import sys
import pytest

# Allow running from project root without packaging setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import database as db  # noqa: E402


@pytest.fixture
def conn(tmp_path):
    """Fresh in-memory-like database for each test (temp file)."""
    db_path = tmp_path / "test_attendance.db"
    connection = db.create_database(str(db_path))
    db.create_tables(connection)
    yield connection
    connection.close()


def test_database_file_is_created(tmp_path):
    db_path = tmp_path / "created.db"
    connection = db.create_database(str(db_path))
    connection.close()
    assert os.path.exists(db_path)


def test_tables_are_created(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    assert {"students", "subjects", "attendance_sessions", "attendance_records"}.issubset(tables)


def test_student_records_are_inserted(conn):
    db.insert_student(conn, "001", "John Snow")
    cur = conn.cursor()
    cur.execute("SELECT name FROM students WHERE student_index = '001'")
    row = cur.fetchone()
    assert row is not None
    assert row[0] == "John Snow"


def test_inserting_same_student_twice_does_not_error(conn):
    db.insert_student(conn, "001", "John Snow")
    db.insert_student(conn, "001", "John Snow")  # should not raise
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM students WHERE student_index = '001'")
    assert cur.fetchone()[0] == 1


def test_attendance_is_saved(conn):
    db.insert_student(conn, "001", "John Snow")
    session_id = db.create_attendance_session(conn, "Math", "2026-07-15")
    saved = db.save_attendance_record(conn, session_id, "001", True, score=0.084)
    assert saved is True

    cur = conn.cursor()
    cur.execute("SELECT present, score FROM attendance_records WHERE student_index = '001'")
    row = cur.fetchone()
    assert row[0] == 1
    assert row[1] == 0.084


def test_attendance_can_be_searched_by_student_index(conn):
    db.insert_student(conn, "001", "John Snow")
    session_id = db.create_attendance_session(conn, "Math", "2026-07-15")
    db.save_attendance_record(conn, session_id, "001", True, score=0.084)

    history = db.get_student_attendance(conn, "001")
    assert len(history) == 1
    assert history[0]["student_name"] == "John Snow"
    assert history[0]["present"] is True


def test_duplicate_attendance_is_prevented(conn):
    db.insert_student(conn, "001", "John Snow")
    session_id = db.create_attendance_session(conn, "Math", "2026-07-15")

    first = db.save_attendance_record(conn, session_id, "001", True, score=0.084)
    second = db.save_attendance_record(conn, session_id, "001", True, score=0.084)

    assert first is True
    assert second is False  # duplicate rejected

    history = db.get_student_attendance(conn, "001")
    assert len(history) == 1


def test_unknown_student_does_not_crash_the_program(conn):
    # No student inserted for "999" — get_student_attendance should just
    # return an empty list, not raise.
    history = db.get_student_attendance(conn, "999")
    assert history == []


def test_multiple_sessions_are_tracked_separately(conn):
    db.insert_student(conn, "001", "John Snow")
    session_1 = db.create_attendance_session(conn, "Math", "2026-07-15")
    session_2 = db.create_attendance_session(conn, "Math", "2026-07-16")

    db.save_attendance_record(conn, session_1, "001", True, score=0.09)
    db.save_attendance_record(conn, session_2, "001", False, score=0.01)

    history = db.get_student_attendance(conn, "001")
    assert len(history) == 2
    assert history[0]["date"] == "2026-07-15"
    assert history[1]["date"] == "2026-07-16"