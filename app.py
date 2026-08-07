from __future__ import annotations

import os
from datetime import date
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from sams import process_attendance
from src.database import create_database, create_tables, get_student_attendance
from src.visualization import (
    calculate_attendance_percentage,
    create_present_absent_chart,
    create_attendance_history_chart,
    create_subject_summary_chart,
)
from src.signature_verification import compare_with_reference_signatures


class AttendanceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Student Attendance Management System")
        self.root.geometry("1180x760")
        self.root.minsize(1000, 680)

        self.current_result = None
        self.preview_photo = None
        self.chart_photo = None

        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")

        title = ttk.Label(
            root,
            text="Student Attendance Management System",
            font=("Segoe UI", 18, "bold"),
        )
        title.pack(pady=(12, 6))

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.process_tab = ttk.Frame(notebook, padding=12)
        self.summary_tab = ttk.Frame(notebook, padding=12)
        self.verify_tab = ttk.Frame(notebook, padding=12)

        notebook.add(self.process_tab, text="Process Attendance")
        notebook.add(self.summary_tab, text="Student Summary")
        notebook.add(self.verify_tab, text="Signature Verification")

        self._build_process_tab()
        self._build_summary_tab()
        self._build_verify_tab()

    # -------------------------------------------------------------
    # Process Attendance tab
    # -------------------------------------------------------------
    def _build_process_tab(self):
        top = ttk.LabelFrame(self.process_tab, text="Input Files", padding=10)
        top.pack(fill="x")
        top.columnconfigure(1, weight=1)

        self.image_var = tk.StringVar(value=str(ROOT / "data" / "signing_sheets" / "1.jpeg"))
        self.xml_var = tk.StringVar(value=str(ROOT / "data" / "info.xml"))
        self.date_var = tk.StringVar(value=date.today().isoformat())

        ttk.Label(top, text="Signing sheet:").grid(row=0, column=0, sticky="w", padx=4, pady=5)
        ttk.Entry(top, textvariable=self.image_var).grid(row=0, column=1, sticky="ew", padx=4, pady=5)
        ttk.Button(top, text="Browse", command=self._browse_image).grid(row=0, column=2, padx=4, pady=5)

        ttk.Label(top, text="Student XML:").grid(row=1, column=0, sticky="w", padx=4, pady=5)
        ttk.Entry(top, textvariable=self.xml_var).grid(row=1, column=1, sticky="ew", padx=4, pady=5)
        ttk.Button(top, text="Browse", command=self._browse_xml).grid(row=1, column=2, padx=4, pady=5)

        ttk.Label(top, text="Session date:").grid(row=2, column=0, sticky="w", padx=4, pady=5)
        ttk.Entry(top, textvariable=self.date_var, width=18).grid(row=2, column=1, sticky="w", padx=4, pady=5)
        ttk.Label(top, text="YYYY-MM-DD").grid(row=2, column=1, sticky="w", padx=(160, 4), pady=5)

        self.process_btn = ttk.Button(top, text="Process Attendance", command=self._process_attendance)
        self.process_btn.grid(row=3, column=0, columnspan=3, pady=(10, 2))

        middle = ttk.Panedwindow(self.process_tab, orient="horizontal")
        middle.pack(fill="both", expand=True, pady=10)

        progress_frame = ttk.LabelFrame(middle, text="Processing Progress", padding=8)
        preview_frame = ttk.LabelFrame(middle, text="Image Processing Preview", padding=8)
        middle.add(progress_frame, weight=1)
        middle.add(preview_frame, weight=2)

        self.progress_text = tk.Text(progress_frame, height=18, width=40, state="disabled", wrap="word")
        self.progress_text.pack(fill="both", expand=True)

        preview_controls = ttk.Frame(preview_frame)
        preview_controls.pack(fill="x", pady=(0, 6))
        ttk.Label(preview_controls, text="Stage:").pack(side="left")
        self.stage_var = tk.StringVar()
        self.stage_combo = ttk.Combobox(preview_controls, textvariable=self.stage_var, state="readonly", width=25)
        self.stage_combo.pack(side="left", padx=6)
        self.stage_combo.bind("<<ComboboxSelected>>", lambda _e: self._show_selected_stage())

        self.preview_label = ttk.Label(preview_frame, anchor="center")
        self.preview_label.pack(fill="both", expand=True)

        results_frame = ttk.LabelFrame(self.process_tab, text="Attendance Results", padding=8)
        results_frame.pack(fill="x")

        columns = ("row", "index", "name", "status", "score")
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=7)
        headings = {
            "row": "Row",
            "index": "Student Index",
            "name": "Student Name",
            "status": "Status",
            "score": "Score",
        }
        widths = {"row": 55, "index": 120, "name": 330, "status": 100, "score": 90}
        for col in columns:
            self.results_tree.heading(col, text=headings[col])
            self.results_tree.column(col, width=widths[col], anchor="center" if col != "name" else "w")
        self.results_tree.pack(fill="x")

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select signing sheet",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
        )
        if path:
            self.image_var.set(path)

    def _browse_xml(self):
        path = filedialog.askopenfilename(
            title="Select student XML",
            filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
        )
        if path:
            self.xml_var.set(path)

    def _append_progress(self, message: str):
        self.progress_text.configure(state="normal")
        self.progress_text.insert("end", message + "\n")
        self.progress_text.see("end")
        self.progress_text.configure(state="disabled")
        self.root.update_idletasks()

    def _clear_progress(self):
        self.progress_text.configure(state="normal")
        self.progress_text.delete("1.0", "end")
        self.progress_text.configure(state="disabled")

    def _process_attendance(self):
        image_path = self.image_var.get().strip()
        xml_path = self.xml_var.get().strip()
        session_date = self.date_var.get().strip()

        if not image_path or not Path(image_path).is_file():
            messagebox.showerror("Missing image", "Please select a valid signing-sheet image.")
            return
        if not xml_path or not Path(xml_path).is_file():
            messagebox.showerror("Missing XML", "Please select a valid info.xml file.")
            return
        if not session_date:
            messagebox.showerror("Missing date", "Please enter the attendance session date.")
            return

        self._clear_progress()
        self.process_btn.configure(state="disabled")
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        try:
            result = process_attendance(
                image_path,
                xml_path,
                session_date=session_date,
                save_to_db=True,
                progress_callback=self._append_progress,
            )
            self.current_result = result

            for row in result["students"]:
                self.results_tree.insert(
                    "",
                    "end",
                    values=(
                        row["row_number"],
                        row["student_index"],
                        row["student_name"],
                        row["status"],
                        f"{row['score']:.4f}",
                    ),
                )

            stage_names = list(result["stages"].keys())
            self.stage_combo["values"] = stage_names
            if stage_names:
                self.stage_var.set(stage_names[-1])
                self._show_selected_stage()

            messagebox.showinfo(
                "Completed",
                f"Attendance processed successfully.\n\nResult saved to:\n{result['result_path']}",
            )
        except Exception as exc:
            self._append_progress(f"ERROR: {exc}")
            messagebox.showerror("Processing error", str(exc))
        finally:
            self.process_btn.configure(state="normal")

    def _show_selected_stage(self):
        if not self.current_result:
            return
        stage = self.stage_var.get()
        path = self.current_result.get("stages", {}).get(stage)
        if path:
            self._display_image(path, self.preview_label, attr_name="preview_photo", max_size=(650, 360))

    # -------------------------------------------------------------
    # Student Summary tab
    # -------------------------------------------------------------
    def _build_summary_tab(self):
        controls = ttk.LabelFrame(self.summary_tab, text="Student Search", padding=10)
        controls.pack(fill="x")

        self.summary_index_var = tk.StringVar()
        ttk.Label(controls, text="Student index:").pack(side="left", padx=4)
        ttk.Entry(controls, textvariable=self.summary_index_var, width=20).pack(side="left", padx=4)
        ttk.Button(controls, text="Generate Summary", command=self._generate_summary).pack(side="left", padx=8)

        self.summary_text = ttk.Label(self.summary_tab, text="Enter a student index to view attendance.", font=("Segoe UI", 11))
        self.summary_text.pack(anchor="w", pady=12)

        chart_controls = ttk.Frame(self.summary_tab)
        chart_controls.pack(fill="x")
        ttk.Label(chart_controls, text="Chart:").pack(side="left")
        self.chart_var = tk.StringVar()
        self.chart_combo = ttk.Combobox(chart_controls, textvariable=self.chart_var, state="readonly", width=28)
        self.chart_combo.pack(side="left", padx=6)
        self.chart_combo.bind("<<ComboboxSelected>>", lambda _e: self._show_selected_chart())

        self.chart_paths = {}
        self.chart_label = ttk.Label(self.summary_tab, anchor="center")
        self.chart_label.pack(fill="both", expand=True, pady=10)

    def _generate_summary(self):
        student_index = self.summary_index_var.get().strip()
        if not student_index:
            messagebox.showerror("Missing student", "Enter a student index.")
            return

        conn = create_database()
        try:
            create_tables(conn)
            records = get_student_attendance(conn, student_index)
        finally:
            conn.close()

        if not records:
            self.summary_text.configure(text=f"No attendance records found for {student_index}.")
            self.chart_paths = {}
            self.chart_combo["values"] = []
            self.chart_label.configure(image="", text="")
            return

        name = records[0].get("student_name", "Unknown Student")
        present = sum(1 for r in records if bool(r.get("present")))
        absent = len(records) - present
        percentage = calculate_attendance_percentage(records)
        self.summary_text.configure(
            text=(
                f"Student: {student_index} - {name}     "
                f"Present: {present}     Absent: {absent}     Attendance: {percentage:.2f}%"
            )
        )

        out_dir = ROOT / "output" / "graphs" / student_index
        self.chart_paths = {
            "Present vs Absent": create_present_absent_chart(records, str(out_dir / "present_absent.png")),
            "Attendance History": create_attendance_history_chart(records, str(out_dir / "attendance_history.png")),
            "Subject Summary": create_subject_summary_chart(records, str(out_dir / "subject_summary.png")),
        }
        names = list(self.chart_paths.keys())
        self.chart_combo["values"] = names
        self.chart_var.set(names[0])
        self._show_selected_chart()

    def _show_selected_chart(self):
        path = self.chart_paths.get(self.chart_var.get())
        if path:
            self._display_image(path, self.chart_label, attr_name="chart_photo", max_size=(820, 500))

    # -------------------------------------------------------------
    # Signature Verification tab
    # -------------------------------------------------------------
    def _build_verify_tab(self):
        frame = ttk.LabelFrame(self.verify_tab, text="Signature Verification", padding=12)
        frame.pack(fill="x")
        frame.columnconfigure(1, weight=1)

        self.verify_index_var = tk.StringVar()
        self.candidate_var = tk.StringVar()
        self.reference_var = tk.StringVar()

        ttk.Label(frame, text="Student index:").grid(row=0, column=0, sticky="w", padx=4, pady=5)
        ttk.Entry(frame, textvariable=self.verify_index_var).grid(row=0, column=1, sticky="ew", padx=4, pady=5)
        ttk.Button(frame, text="Use Default References", command=self._set_default_refs).grid(row=0, column=2, padx=4, pady=5)

        ttk.Label(frame, text="Candidate signature:").grid(row=1, column=0, sticky="w", padx=4, pady=5)
        ttk.Entry(frame, textvariable=self.candidate_var).grid(row=1, column=1, sticky="ew", padx=4, pady=5)
        ttk.Button(frame, text="Browse", command=self._browse_candidate).grid(row=1, column=2, padx=4, pady=5)

        ttk.Label(frame, text="Reference folder:").grid(row=2, column=0, sticky="w", padx=4, pady=5)
        ttk.Entry(frame, textvariable=self.reference_var).grid(row=2, column=1, sticky="ew", padx=4, pady=5)
        ttk.Button(frame, text="Browse", command=self._browse_reference_dir).grid(row=2, column=2, padx=4, pady=5)

        ttk.Button(frame, text="Verify Signature", command=self._verify_signature).grid(row=3, column=0, columnspan=3, pady=12)

        self.verify_result = ttk.Label(
            self.verify_tab,
            text="Select a candidate signature and reference folder.",
            font=("Segoe UI", 12),
            justify="left",
        )
        self.verify_result.pack(anchor="w", pady=18)

    def _set_default_refs(self):
        index = self.verify_index_var.get().strip()
        if not index:
            messagebox.showerror("Missing student", "Enter the student index first.")
            return
        self.reference_var.set(str(ROOT / "data" / "reference_signatures" / index))

    def _browse_candidate(self):
        path = filedialog.askopenfilename(
            title="Select candidate signature",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")],
        )
        if path:
            self.candidate_var.set(path)

    def _browse_reference_dir(self):
        path = filedialog.askdirectory(title="Select reference-signature folder")
        if path:
            self.reference_var.set(path)

    def _verify_signature(self):
        candidate = self.candidate_var.get().strip()
        reference_dir = self.reference_var.get().strip()

        if not Path(candidate).is_file():
            messagebox.showerror("Missing candidate", "Select a valid candidate signature image.")
            return
        if not Path(reference_dir).is_dir():
            messagebox.showerror("Missing references", "Select a valid reference-signature folder.")
            return

        result = compare_with_reference_signatures(candidate, reference_dir, threshold=0.65)
        self.verify_result.configure(
            text=(
                f"Decision: {result['decision']}\n"
                f"Best similarity score: {result['best_score']:.2%}\n"
                f"Best reference: {result['best_reference'] or 'N/A'}\n"
                f"References compared: {result['reference_count']}"
            )
        )

    # -------------------------------------------------------------
    # Shared image helper
    # -------------------------------------------------------------
    def _display_image(self, path, label, attr_name: str, max_size=(700, 450)):
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail(max_size)
            photo = ImageTk.PhotoImage(image)
            setattr(self, attr_name, photo)
            label.configure(image=photo, text="")
        except Exception as exc:
            label.configure(image="", text=f"Could not preview image:\n{exc}")


def main():
    root = tk.Tk()
    AttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
