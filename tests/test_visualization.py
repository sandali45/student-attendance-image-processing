"""Tests for visualization module and infovis CLI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

import infovis  # noqa: E402
from src import visualization  # noqa: E402


@pytest.fixture
def sample_records() -> list[dict]:
	"""Return representative attendance records for one student."""
	return [
		{"date": "2026-07-10", "subject": "CS402.3", "present": True},
		{"date": "2026-07-17", "subject": "CS402.3", "present": False},
		{"date": "2026-07-24", "subject": "CS402.3", "present": True},
		{"date": "2026-07-31", "subject": "CS405.2", "present": True},
	]


def test_attendance_percentage_calculates_correctly(sample_records: list[dict]) -> None:
	"""Attendance percentage should be present/total * 100 with 2 decimals."""
	assert visualization.calculate_attendance_percentage(sample_records) == 75.0


def test_present_and_absent_totals_are_correct(sample_records: list[dict]) -> None:
	"""Present/absent counters should match the input records."""
	present_count, absent_count = visualization._count_present_absent(sample_records)
	assert present_count == 3
	assert absent_count == 1


def test_graph_files_are_created(sample_records: list[dict], tmp_path: Path) -> None:
	"""Chart functions should write PNG files to disk."""
	present_absent = tmp_path / "present_absent.png"
	history = tmp_path / "attendance_history.png"
	subject = tmp_path / "subject_summary.png"

	visualization.create_present_absent_chart(sample_records, str(present_absent))
	visualization.create_attendance_history_chart(sample_records, str(history))
	visualization.create_subject_summary_chart(sample_records, str(subject))

	assert os.path.exists(present_absent)
	assert os.path.exists(history)
	assert os.path.exists(subject)


def test_history_data_is_sorted_chronologically() -> None:
	"""History data preparation should return records in ascending date order."""
	unsorted_records = [
		{"date": "2026-07-24", "subject": "CS402.3", "present": True},
		{"date": "2026-07-10", "subject": "CS402.3", "present": True},
		{"date": "2026-07-17", "subject": "CS402.3", "present": False},
	]

	sorted_records = visualization._prepare_history_data(unsorted_records)
	sorted_dates = [record["date"] for record in sorted_records]
	assert sorted_dates == ["2026-07-10", "2026-07-17", "2026-07-24"]


def test_empty_attendance_history_is_handled_without_exceptions(
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
	tmp_path: Path,
) -> None:
	"""Empty history should not raise and should print a helpful message."""
	monkeypatch.setattr(visualization, "OUTPUT_GRAPHS_ROOT", tmp_path)
	visualization.display_student_summary("001", "John Snow", [])
	output = capsys.readouterr().out
	assert "No attendance records found" in output


def test_invalid_student_index_prints_useful_message_and_exits_non_zero(
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	"""CLI should fail gracefully for unknown student index."""

	def mock_get_student_attendance(student_index: str) -> list[dict]:
		_ = student_index
		return []

	monkeypatch.setattr(infovis.database, "get_student_attendance", mock_get_student_attendance)
	monkeypatch.setattr(infovis, "_load_student_name_from_xml", lambda student_index: None)

	exit_code = infovis.main(["999"])
	output = capsys.readouterr().out

	assert exit_code == 1
	assert "No student found with index '999'" in output

