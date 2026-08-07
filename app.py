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


# ------------------------------------------------------------
# Simple UI colours
# ------------------------------------------------------------
BG = "#F4F6FA"
CARD = "#FFFFFF"
TEXT = "#1F2937"
MUTED = "#6B7280"
ACCENT = "#4F46E5"
ACCENT_DARK = "#4338CA"
GREEN = "#15803D"
GREEN_BG = "#ECFDF3"
RED = "#B42318"
RED_BG = "#FEF3F2"
BORDER = "#E5E7EB"


class AttendanceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Student Attendance Management System")
        self.root.geometry("1120x740")
        self.root.minsize(980, 660)
        self.root.configure(bg=BG)

        self.current_result = None
        self.preview_photo = None
        self.chart_photo = None
        self.chart_paths = {}

        self._setup_styles()
        self._build_header()
        self._build_tabs()

    # --------------------------------------------------------
    # Styling
    # --------------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style()

        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("App.TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)

        style.configure(
            "TLabel",
            background=BG,
            foreground=TEXT,
            font=("Segoe UI", 10),
        )

        style.configure(
            "Card.TLabel",
            background=CARD,
            foreground=TEXT,
            font=("Segoe UI", 10),
        )

        style.configure(
            "Muted.Card.TLabel",
            background=CARD,
            foreground=MUTED,
            font=("Segoe UI", 9),
        )

        style.configure(
            "Title.TLabel",
            background=BG,
            foreground=TEXT,
            font=("Segoe UI", 22, "bold"),
        )

        style.configure(
            "Subtitle.TLabel",
            background=BG,
            foreground=MUTED,
            font=("Segoe UI", 10),
        )

        style.configure(
            "Section.Card.TLabel",
            background=CARD,
            foreground=TEXT,
            font=("Segoe UI", 12, "bold"),
        )

        style.configure(
            "Metric.Card.TLabel",
            background=CARD,
            foreground=TEXT,
            font=("Segoe UI", 16, "bold"),
        )

        style.configure(
            "Accent.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(18, 10),
            background=ACCENT,
            foreground="white",
            borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", ACCENT_DARK), ("disabled", "#A5B4FC")],
            foreground=[("disabled", "#F9FAFB")],
        )

        style.configure(
            "Soft.TButton",
            font=("Segoe UI", 9),
            padding=(10, 7),
            background="#EEF2FF",
            foreground=ACCENT_DARK,
            borderwidth=0,
        )
        style.map("Soft.TButton", background=[("active", "#E0E7FF")])

        style.configure(
            "TNotebook",
            background=BG,
            borderwidth=0,
            tabmargins=(0, 0, 0, 0),
        )
        style.configure(
            "TNotebook.Tab",
            font=("Segoe UI", 10, "bold"),
            padding=(18, 10),
            background="#EDEFF5",
            foreground=MUTED,
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", CARD)],
            foreground=[("selected", ACCENT)],
        )

        style.configure(
            "Treeview",
            font=("Segoe UI", 9),
            rowheight=30,
            background=CARD,
            fieldbackground=CARD,
            foreground=TEXT,
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            font=("Segoe UI", 9, "bold"),
            background="#F8FAFC",
            foreground=TEXT,
            relief="flat",
            padding=(6, 7),
        )
        style.map("Treeview", background=[("selected", "#E0E7FF")])

        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#E5E7EB",
            background=ACCENT,
            bordercolor="#E5E7EB",
            lightcolor=ACCENT,
            darkcolor=ACCENT,
        )

    # --------------------------------------------------------
    # Header + tabs
    # --------------------------------------------------------
    def _build_header(self):
        header = ttk.Frame(self.root, style="App.TFrame")
        header.pack(fill="x", padx=24, pady=(20, 12))

        left = ttk.Frame(header, style="App.TFrame")
        left.pack(side="left")

        ttk.Label(
            left,
            text="Student Attendance System",
            style="Title.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            left,
            text="Process signing sheets, view attendance and verify signatures.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

    def _build_tabs(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=24, pady=(0, 22))

        self.process_tab = ttk.Frame(notebook, style="App.TFrame", padding=14)
        self.summary_tab = ttk.Frame(notebook, style="App.TFrame", padding=14)

        notebook.add(self.process_tab, text="Process Attendance")
        notebook.add(self.summary_tab, text="Student Summary")

        self._build_process_tab()
        self._build_summary_tab()

    # --------------------------------------------------------
    # Small reusable pieces
    # --------------------------------------------------------
    def _card(self, parent):
        outer = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        return outer

    def _entry_row(self, parent, label_text, variable, browse_command=None, row=0, hint=None):
        tk.Label(
            parent,
            text=label_text,
            bg=CARD,
            fg=TEXT,
            font=("Segoe UI", 9, "bold"),
        ).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=7)

        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=7)

        if browse_command:
            ttk.Button(
                parent,
                text="Browse",
                style="Soft.TButton",
                command=browse_command,
            ).grid(row=row, column=2, padx=(8, 0), pady=7)

        if hint:
            tk.Label(
                parent,
                text=hint,
                bg=CARD,
                fg=MUTED,
                font=("Segoe UI", 8),
            ).grid(row=row + 1, column=1, sticky="w", pady=(0, 3))

    # --------------------------------------------------------
    # Process Attendance
    # --------------------------------------------------------
    def _build_process_tab(self):
        self.image_var = tk.StringVar(
            value=str(ROOT / "data" / "signing_sheets" / "1.jpeg")
        )
        self.xml_var = tk.StringVar(value=str(ROOT / "data" / "info.xml"))
        self.date_var = tk.StringVar(value=date.today().isoformat())

        # Input card
        input_card = self._card(self.process_tab)
        input_card.pack(fill="x", pady=(0, 12))
        input_card.grid_columnconfigure(0, weight=1)

        content = tk.Frame(input_card, bg=CARD)
        content.pack(fill="x", padx=18, pady=16)
        content.grid_columnconfigure(1, weight=1)

        tk.Label(
            content,
            text="Choose attendance files",
            bg=CARD,
            fg=TEXT,
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

        tk.Label(
            content,
            text="Select one signing-sheet image and the matching student XML file.",
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 8))

        self._entry_row(
            content,
            "Signing sheet",
            self.image_var,
            self._browse_image,
            row=2,
        )
        self._entry_row(
            content,
            "Student XML",
            self.xml_var,
            self._browse_xml,
            row=3,
        )

        tk.Label(
            content,
            text="Session date",
            bg=CARD,
            fg=TEXT,
            font=("Segoe UI", 9, "bold"),
        ).grid(row=4, column=0, sticky="w", padx=(0, 10), pady=7)

        ttk.Entry(content, textvariable=self.date_var, width=18).grid(
            row=4, column=1, sticky="w", pady=7
        )

        self.process_btn = ttk.Button(
            content,
            text="Process Attendance",
            style="Accent.TButton",
            command=self._process_attendance,
        )
        self.process_btn.grid(row=5, column=0, columnspan=3, pady=(12, 2))

        # Simple status area
        status_card = self._card(self.process_tab)
        status_card.pack(fill="x", pady=(0, 12))

        status_inner = tk.Frame(status_card, bg=CARD)
        status_inner.pack(fill="x", padx=18, pady=12)

        self.status_label = tk.Label(
            status_inner,
            text="Ready to process a signing sheet.",
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 9),
            anchor="w",
        )
        self.status_label.pack(fill="x")

        self.progress = ttk.Progressbar(
            status_inner,
            mode="indeterminate",
            style="Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", pady=(8, 0))

        # Main result area
        body = ttk.Panedwindow(self.process_tab, orient="horizontal")
        body.pack(fill="both", expand=True)

        preview_card = self._card(body)
        results_card = self._card(body)
        body.add(preview_card, weight=1)
        body.add(results_card, weight=1)

        # Preview
        preview_top = tk.Frame(preview_card, bg=CARD)
        preview_top.pack(fill="x", padx=16, pady=(14, 8))

        tk.Label(
            preview_top,
            text="Image Preview",
            bg=CARD,
            fg=TEXT,
            font=("Segoe UI", 12, "bold"),
        ).pack(side="left")

        self.stage_var = tk.StringVar(value="Original")
        self.stage_combo = ttk.Combobox(
            preview_top,
            textvariable=self.stage_var,
            state="readonly",
            width=18,
        )
        self.stage_combo.pack(side="right")
        self.stage_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._show_selected_stage(),
        )

        preview_box = tk.Frame(preview_card, bg="#FAFAFB")
        preview_box.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.preview_label = tk.Label(
            preview_box,
            text="The processed image will appear here.",
            bg="#FAFAFB",
            fg=MUTED,
            font=("Segoe UI", 9),
        )
        self.preview_label.pack(fill="both", expand=True, padx=8, pady=8)

        # Results
        results_top = tk.Frame(results_card, bg=CARD)
        results_top.pack(fill="x", padx=16, pady=(14, 8))

        tk.Label(
            results_top,
            text="Attendance Results",
            bg=CARD,
            fg=TEXT,
            font=("Segoe UI", 12, "bold"),
        ).pack(side="left")

        self.result_count_label = tk.Label(
            results_top,
            text="No results yet",
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 9),
        )
        self.result_count_label.pack(side="right")

        tree_frame = tk.Frame(results_card, bg=CARD)
        tree_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        columns = ("index", "name", "status", "score")
        self.results_tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            height=12,
        )

        self.results_tree.heading("index", text="Student Index")
        self.results_tree.heading("name", text="Student Name")
        self.results_tree.heading("status", text="Status")
        self.results_tree.heading("score", text="Score")

        self.results_tree.column("index", width=105, anchor="center")
        self.results_tree.column("name", width=220, anchor="w")
        self.results_tree.column("status", width=80, anchor="center")
        self.results_tree.column("score", width=70, anchor="center")

        self.results_tree.tag_configure("present", background=GREEN_BG, foreground=GREEN)
        self.results_tree.tag_configure("absent", background=RED_BG, foreground=RED)

        scroll = ttk.Scrollbar(
            tree_frame,
            orient="vertical",
            command=self.results_tree.yview,
        )
        self.results_tree.configure(yscrollcommand=scroll.set)

        self.results_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # Show initial image if it exists
        if Path(self.image_var.get()).is_file():
            self._display_image(
                self.image_var.get(),
                self.preview_label,
                "preview_photo",
                max_size=(460, 390),
            )

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select signing sheet",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.image_var.set(path)
            self.stage_var.set("Original")
            self._display_image(
                path,
                self.preview_label,
                "preview_photo",
                max_size=(460, 390),
            )

    def _browse_xml(self):
        path = filedialog.askopenfilename(
            title="Select student XML",
            filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
        )
        if path:
            self.xml_var.set(path)

    def _set_status(self, message, colour=MUTED):
        self.status_label.configure(text=message, fg=colour)
        self.root.update_idletasks()

    def _process_attendance(self):
        image_path = self.image_var.get().strip()
        xml_path = self.xml_var.get().strip()
        session_date = self.date_var.get().strip()

        if not Path(image_path).is_file():
            messagebox.showerror(
                "Missing image",
                "Please select a valid signing-sheet image.",
            )
            return

        if not Path(xml_path).is_file():
            messagebox.showerror(
                "Missing XML",
                "Please select a valid info.xml file.",
            )
            return

        if not session_date:
            messagebox.showerror(
                "Missing date",
                "Please enter the attendance session date.",
            )
            return

        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        self.process_btn.configure(state="disabled")
        self.progress.start(10)
        self._set_status("Processing attendance...")

        try:
            result = process_attendance(
                image_path,
                xml_path,
                session_date=session_date,
                save_to_db=True,
                progress_callback=lambda message: self._set_status(message),
            )

            self.current_result = result

            present_count = 0
            for row in result["students"]:
                if row["present"]:
                    present_count += 1

                tag = "present" if row["present"] else "absent"

                self.results_tree.insert(
                    "",
                    "end",
                    values=(
                        row["student_index"],
                        row["student_name"],
                        row["status"],
                        f"{row['score']:.4f}",
                    ),
                    tags=(tag,),
                )

            total = len(result["students"])
            absent_count = total - present_count

            self.result_count_label.configure(
                text=f"{present_count} Present  •  {absent_count} Absent"
            )

            stage_names = list(result.get("stages", {}).keys())
            self.stage_combo["values"] = stage_names

            if stage_names:
                self.stage_var.set("Binary" if "Binary" in stage_names else stage_names[-1])
                self._show_selected_stage()

            self._set_status(
                f"Completed successfully • {total} students processed",
                GREEN,
            )

        except Exception as exc:
            self._set_status(f"Error: {exc}", RED)
            messagebox.showerror("Processing error", str(exc))

        finally:
            self.progress.stop()
            self.process_btn.configure(state="normal")

    def _show_selected_stage(self):
        if not self.current_result:
            if self.stage_var.get() == "Original" and Path(self.image_var.get()).is_file():
                self._display_image(
                    self.image_var.get(),
                    self.preview_label,
                    "preview_photo",
                    max_size=(460, 390),
                )
            return

        stage = self.stage_var.get()
        path = self.current_result.get("stages", {}).get(stage)

        if path:
            self._display_image(
                path,
                self.preview_label,
                "preview_photo",
                max_size=(460, 390),
            )

    # --------------------------------------------------------
    # Student Summary
    # --------------------------------------------------------
    def _build_summary_tab(self):
        top_card = self._card(self.summary_tab)
        top_card.pack(fill="x", pady=(0, 12))

        top = tk.Frame(top_card, bg=CARD)
        top.pack(fill="x", padx=18, pady=16)

        tk.Label(
            top,
            text="Student Attendance Summary",
            bg=CARD,
            fg=TEXT,
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w")

        tk.Label(
            top,
            text="Enter a student index to view the attendance summary.",
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 10))

        search = tk.Frame(top, bg=CARD)
        search.pack(fill="x")

        self.summary_index_var = tk.StringVar()

        ttk.Entry(
            search,
            textvariable=self.summary_index_var,
            width=28,
        ).pack(side="left")

        ttk.Button(
            search,
            text="View Summary",
            style="Accent.TButton",
            command=self._generate_summary,
        ).pack(side="left", padx=(10, 0))

        metrics = tk.Frame(self.summary_tab, bg=BG)
        metrics.pack(fill="x", pady=(0, 12))

        self.metric_name = self._metric_card(metrics, "Student", "—")
        self.metric_present = self._metric_card(metrics, "Present", "0")
        self.metric_absent = self._metric_card(metrics, "Absent", "0")
        self.metric_rate = self._metric_card(metrics, "Attendance", "0%")

        chart_card = self._card(self.summary_tab)
        chart_card.pack(fill="both", expand=True)

        chart_top = tk.Frame(chart_card, bg=CARD)
        chart_top.pack(fill="x", padx=16, pady=(14, 8))

        tk.Label(
            chart_top,
            text="Attendance Chart",
            bg=CARD,
            fg=TEXT,
            font=("Segoe UI", 12, "bold"),
        ).pack(side="left")

        self.chart_var = tk.StringVar()
        self.chart_combo = ttk.Combobox(
            chart_top,
            textvariable=self.chart_var,
            state="readonly",
            width=22,
        )
        self.chart_combo.pack(side="right")
        self.chart_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._show_selected_chart(),
        )

        chart_box = tk.Frame(chart_card, bg="#FAFAFB")
        chart_box.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.chart_label = tk.Label(
            chart_box,
            text="No attendance chart yet.",
            bg="#FAFAFB",
            fg=MUTED,
            font=("Segoe UI", 9),
        )
        self.chart_label.pack(fill="both", expand=True, padx=8, pady=8)

    def _metric_card(self, parent, title, value):
        card = tk.Frame(
            parent,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        card.pack(side="left", fill="x", expand=True, padx=(0, 8))

        tk.Label(
            card,
            text=title,
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=14, pady=(12, 2))

        value_label = tk.Label(
            card,
            text=value,
            bg=CARD,
            fg=TEXT,
            font=("Segoe UI", 15, "bold"),
        )
        value_label.pack(anchor="w", padx=14, pady=(0, 12))
        return value_label

    def _generate_summary(self):
        student_index = self.summary_index_var.get().strip()

        if not student_index:
            messagebox.showerror(
                "Missing student",
                "Enter a student index.",
            )
            return

        conn = create_database()
        try:
            create_tables(conn)
            records = get_student_attendance(conn, student_index)
        finally:
            conn.close()

        if not records:
            messagebox.showinfo(
                "No records",
                f"No attendance records found for {student_index}.",
            )
            self.metric_name.configure(text="—")
            self.metric_present.configure(text="0")
            self.metric_absent.configure(text="0")
            self.metric_rate.configure(text="0%")
            return

        name = records[0].get("student_name", "Unknown Student")
        present = sum(1 for r in records if bool(r.get("present")))
        absent = len(records) - present
        percentage = calculate_attendance_percentage(records)

        self.metric_name.configure(text=name)
        self.metric_present.configure(text=str(present))
        self.metric_absent.configure(text=str(absent))
        self.metric_rate.configure(text=f"{percentage:.1f}%")

        out_dir = ROOT / "output" / "graphs" / student_index

        self.chart_paths = {
            "Present vs Absent": create_present_absent_chart(
                records,
                str(out_dir / "present_absent.png"),
            ),
            "Attendance History": create_attendance_history_chart(
                records,
                str(out_dir / "attendance_history.png"),
            ),
            "Subject Summary": create_subject_summary_chart(
                records,
                str(out_dir / "subject_summary.png"),
            ),
        }

        names = list(self.chart_paths.keys())
        self.chart_combo["values"] = names
        self.chart_var.set(names[0])
        self._show_selected_chart()

    def _show_selected_chart(self):
        path = self.chart_paths.get(self.chart_var.get())

        if path:
            self._display_image(
                path,
                self.chart_label,
                "chart_photo",
                max_size=(800, 430),
            )

    # --------------------------------------------------------
    # Image display helper
    # --------------------------------------------------------
    def _display_image(self, path, label, attr_name, max_size=(700, 450)):
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail(max_size)

            photo = ImageTk.PhotoImage(image)
            setattr(self, attr_name, photo)

            label.configure(
                image=photo,
                text="",
            )

        except Exception as exc:
            label.configure(
                image="",
                text=f"Could not preview image.\n{exc}",
            )


def main():
    root = tk.Tk()
    AttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
