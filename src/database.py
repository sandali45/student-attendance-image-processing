"""
src/database.py
Member 8 — SQLite database creation, saving and searching.

Tables:
    students             (student_index PK, name)
    subjects             (subject_id PK, subject_name)
    attendance_sessions  (session_id PK, subject_id FK, date, sheet_filename)
    attendance_records   (record_id PK, session_id FK, student_index FK,
                          present, score, UNIQUE(session_id, student_index))
"""

import sqlite3
import os


DEFAULT_DB_PATH = os.path.join("database", "attendance.db")


def create_database(db_path=DEFAULT_DB_PATH):
    """Create (or open) the SQLite database file and return a connection."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_tables(conn):
    """Create all required tables if they don't already exist."""
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_index TEXT PRIMARY KEY,
            name TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id INTEGER,
            date TEXT NOT NULL,
            sheet_filename TEXT,
            FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance_records (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            student_index TEXT NOT NULL,
            present INTEGER NOT NULL,
            score REAL,
            FOREIGN KEY (session_id) REFERENCES attendance_sessions(session_id),
            FOREIGN KEY (student_index) REFERENCES students(student_index),
            UNIQUE(session_id, student_index)
        )
    """)

    conn.commit()


def insert_student(conn, student_index, name):
    """Insert a student. Ignores if the student already exists."""
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO students (student_index, name)
        VALUES (?, ?)
    """, (student_index, name))
    conn.commit()


def create_attendance_session(conn, subject_name, date, sheet_filename=None):
    """
    Create (or reuse) a subject, then create a new attendance session
    for that subject/date and return the session_id.
    """
    cur = conn.cursor()

    cur.execute("SELECT subject_id FROM subjects WHERE subject_name = ?", (subject_name,))
    row = cur.fetchone()
    if row:
        subject_id = row[0]
    else:
        cur.execute("INSERT INTO subjects (subject_name) VALUES (?)", (subject_name,))
        subject_id = cur.lastrowid

    cur.execute("""
        INSERT INTO attendance_sessions (subject_id, date, sheet_filename)
        VALUES (?, ?, ?)
    """, (subject_id, date, sheet_filename))
    conn.commit()

    return cur.lastrowid


def save_attendance_record(conn, session_id, student_index, present, score=None):
    """
    Save one attendance record. Returns True if saved,
    False if a record for this student+session already exists (duplicate).
    """
    if prevent_duplicate_records(conn, session_id, student_index):
        return False

    cur = conn.cursor()
    cur.execute("""
        INSERT INTO attendance_records (session_id, student_index, present, score)
        VALUES (?, ?, ?, ?)
    """, (session_id, student_index, int(bool(present)), score))
    conn.commit()
    return True


def prevent_duplicate_records(conn, session_id, student_index):
    """Return True if a record already exists for this session+student."""
    cur = conn.cursor()
    cur.execute("""
        SELECT 1 FROM attendance_records
        WHERE session_id = ? AND student_index = ?
    """, (session_id, student_index))
    return cur.fetchone() is not None


def get_student_attendance(conn, student_index):
    """
    Return all attendance records for a student, joined with session
    date and subject name, ordered by date.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT s.name, sub.subject_name, sess.date, ar.present, ar.score
        FROM attendance_records ar
        JOIN attendance_sessions sess ON ar.session_id = sess.session_id
        JOIN students s ON ar.student_index = s.student_index
        LEFT JOIN subjects sub ON sess.subject_id = sub.subject_id
        WHERE ar.student_index = ?
        ORDER BY sess.date
    """, (student_index,))
    rows = cur.fetchall()

    results = []
    for name, subject_name, date, present, score in rows:
        results.append({
            "student_name": name,
            "subject": subject_name,
            "date": date,
            "present": bool(present),
            "score": score,
        })
    return results


if __name__ == "__main__":
    # Quick manual smoke test using fake data, no dependency on other members.
    conn = create_database("database/attendance_demo.db")
    create_tables(conn)

    insert_student(conn, "001", "John Snow")
    insert_student(conn, "007", "James Bond")
    insert_student(conn, "009", "Andare")

    session_id = create_attendance_session(conn, "Math", "2026-07-15", "sheet_01.png")

    save_attendance_record(conn, session_id, "001", True, score=0.084)
    save_attendance_record(conn, session_id, "007", True, score=0.067)
    save_attendance_record(conn, session_id, "009", False, score=0.004)

    # duplicate attempt should be rejected
    duplicate_ok = save_attendance_record(conn, session_id, "001", True, score=0.084)
    print("Duplicate insert accepted?", duplicate_ok)  # should print False

    print("Attendance for 001:", get_student_attendance(conn, "001"))
    conn.close()