"""Visualization utilities for attendance summaries and charts.

Assumption for integration:
Each attendance record is expected to include at least the keys
"date" (ISO-like string), "subject" (string or None), and "present"
(bool-like value: bool or 0/1).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)
OUTPUT_GRAPHS_ROOT = Path("output") / "graphs"


class VisualizationError(Exception):
	"""Raised when a chart cannot be created or saved."""


def _count_present_absent(records: list[dict]) -> tuple[int, int]:
	"""Return present and absent counts from attendance records."""
	present_count = sum(1 for record in records if bool(record.get("present", False)))
	absent_count = len(records) - present_count
	return present_count, absent_count


def _parse_date(value: str) -> datetime:
	"""Parse a date string using ISO style first, then common fallbacks."""
	for fmt in (None, "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
		try:
			if fmt is None:
				return datetime.fromisoformat(value)
			return datetime.strptime(value, fmt)
		except (TypeError, ValueError):
			continue
	raise ValueError(f"Unsupported date format: {value!r}")


def _prepare_history_data(records: list[dict]) -> list[dict]:
	"""Return records sorted chronologically for history plotting."""
	sortable: list[tuple[datetime, dict]] = []
	fallback: list[dict] = []

	for record in records:
		date_value = str(record.get("date", ""))
		try:
			sortable.append((_parse_date(date_value), record))
		except ValueError:
			fallback.append(record)

	sortable.sort(key=lambda item: item[0])
	sorted_records = [item[1] for item in sortable]

	if fallback:
		LOGGER.warning("Some records had unsupported dates and were appended unsorted.")
		sorted_records.extend(fallback)

	return sorted_records


def calculate_attendance_percentage(records: list[dict]) -> float:
	"""Calculate attendance percentage rounded to two decimals.

	Args:
		records: Attendance records containing a "present" value.

	Returns:
		Percentage value in range 0-100. Returns 0.0 when records are empty.
	"""
	if not records:
		return 0.0

	present_count, _ = _count_present_absent(records)
	percentage = (present_count / len(records)) * 100
	return round(percentage, 2)


def _save_current_figure(output_path: str) -> str:
	"""Save the active matplotlib figure and return the output path string."""
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)

	try:
		plt.tight_layout()
		plt.savefig(output, format="png", dpi=150)
		LOGGER.info("Chart saved: %s", output)
		return str(output)
	except Exception as exc:  # pragma: no cover - defensive wrapper
		raise VisualizationError(f"Failed to save chart to {output}") from exc
	finally:
		plt.close()


def create_present_absent_chart(records: list[dict], output_path: str) -> str:
	"""Create and save a present-vs-absent chart.

	Args:
		records: Attendance records for one student.
		output_path: PNG path for chart output.

	Returns:
		The saved chart path.
	"""
	present_count, absent_count = _count_present_absent(records)

	plt.figure(figsize=(7, 4))
	labels = ["Present", "Absent"]
	values = [present_count, absent_count]
	colors = ["#2E8B57", "#B22222"]
	plt.bar(labels, values, color=colors)
	plt.title("Attendance Summary: Present vs Absent")
	plt.ylabel("Session Count")

	return _save_current_figure(output_path)


def create_attendance_history_chart(records: list[dict], output_path: str) -> str:
	"""Create and save a chronological attendance history chart.

	Args:
		records: Attendance records for one student.
		output_path: PNG path for chart output.

	Returns:
		The saved chart path.
	"""
	sorted_records = _prepare_history_data(records)

	plt.figure(figsize=(9, 4))
	x_positions = list(range(1, len(sorted_records) + 1))
	y_values = [1 if bool(record.get("present", False)) else 0 for record in sorted_records]
	colors = ["#2E8B57" if value == 1 else "#B22222" for value in y_values]

	plt.scatter(x_positions, y_values, c=colors, s=80)
	plt.yticks([0, 1], ["Absent", "Present"])
	plt.xticks(x_positions, [record.get("date", "") for record in sorted_records], rotation=45, ha="right")
	plt.title("Attendance History")
	plt.xlabel("Date")
	plt.ylabel("Status")

	return _save_current_figure(output_path)


def create_subject_summary_chart(records: list[dict], output_path: str) -> str:
	"""Create and save per-subject attendance percentage chart.

	Args:
		records: Attendance records for one student.
		output_path: PNG path for chart output.

	Returns:
		The saved chart path.
	"""
	subject_buckets: dict[str, list[dict]] = defaultdict(list)
	for record in records:
		subject = record.get("subject") or "Unknown"
		subject_buckets[str(subject)].append(record)

	if not subject_buckets:
		subject_buckets["No Data"] = []

	subjects = list(subject_buckets.keys())
	percentages = [calculate_attendance_percentage(subject_buckets[subject]) for subject in subjects]

	plt.figure(figsize=(8, 4))
	plt.bar(subjects, percentages, color="#1F77B4")
	plt.ylim(0, 100)
	plt.ylabel("Attendance (%)")
	plt.title("Attendance by Subject")
	plt.xticks(rotation=20, ha="right")

	return _save_current_figure(output_path)


def display_student_summary(student_index: str, student_name: str, records: list[dict]) -> None:
	"""Print student summary and generate all required charts.

	Args:
		student_index: Student index identifier.
		student_name: Student display name.
		records: Attendance records for this student.
	"""
	present_count, absent_count = _count_present_absent(records)
	percentage = calculate_attendance_percentage(records)

	print(f"Student Index: {student_index}")
	print(f"Student Name : {student_name}")
	print(f"Attendance   : {percentage:.2f}%")
	print(f"Present      : {present_count}")
	print(f"Absent       : {absent_count}")

	if not records:
		print("No attendance records found")
		return

	output_dir = OUTPUT_GRAPHS_ROOT / student_index
	output_dir.mkdir(parents=True, exist_ok=True)

	create_present_absent_chart(records, str(output_dir / "present_absent.png"))
	create_attendance_history_chart(records, str(output_dir / "attendance_history.png"))
	create_subject_summary_chart(records, str(output_dir / "subject_summary.png"))

