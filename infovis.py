"""CLI entry point for attendance visualization.

Usage:
	python infovis.py 001
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

import database  # noqa: E402
from visualization import display_student_summary  # noqa: E402

try:
	import xml_parser  # noqa: E402
except Exception:  # pragma: no cover - defensive optional dependency import
	xml_parser = None


def _load_student_name_from_xml(student_index: str) -> str | None:
	"""Try resolving student name from XML mapping; return None if unavailable."""
	if xml_parser is None:
		return None

	xml_path = ROOT / "data" / "info.xml"
	if not xml_path.exists():
		return None

	try:
		root = xml_parser.read_xml_file(str(xml_path))
		records = xml_parser.extract_student_records(root)
	except Exception:
		return None

	for record in records:
		if str(record.get("student_index")) == student_index:
			name = record.get("student_name")
			return str(name) if name else None
	return None


def _get_records_and_name_from_database(student_index: str) -> tuple[list[dict], str | None]:
	"""Fetch attendance records and fallback student name from database module."""
	get_fn = getattr(database, "get_student_attendance", None)
	if get_fn is None:
		raise RuntimeError("database.get_student_attendance is not available")

	signature = inspect.signature(get_fn)
	param_count = len(signature.parameters)

	if param_count == 1:
		records = get_fn(student_index)
	elif param_count >= 2:
		conn = database.create_database()
		try:
			if hasattr(database, "create_tables"):
				database.create_tables(conn)
			records = get_fn(conn, student_index)
		finally:
			conn.close()
	else:
		raise RuntimeError("Unsupported get_student_attendance signature")

	records = records or []
	db_name = None
	if records:
		first = records[0]
		db_name = first.get("student_name")
	return records, (str(db_name) if db_name else None)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	"""Parse CLI arguments."""
	parser = argparse.ArgumentParser(description="Generate student attendance visualizations")
	parser.add_argument("student_index", help="Student index (e.g. 001)")
	return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
	"""Run attendance visualization CLI and return process exit code."""
	args = _parse_args(argv)
	student_index = args.student_index.strip()

	try:
		records, db_name = _get_records_and_name_from_database(student_index)
	except Exception as exc:
		print(f"Error: Could not retrieve attendance data ({exc}).")
		return 1

	xml_name = _load_student_name_from_xml(student_index)
	student_name = xml_name or db_name

	if student_name is None and not records:
		print(f"Error: No student found with index '{student_index}'.")
		return 1

	if student_name is None:
		student_name = "Unknown Student"

	display_student_summary(student_index, student_name, records)
	return 0


if __name__ == "__main__":
	sys.exit(main())

