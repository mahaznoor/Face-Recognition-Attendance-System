"""
gui.py
------
Tkinter-based graphical user interface for the Face Recognition
Attendance System.

Windows:
    - LoginScreen           : simple gatekeeper screen
    - DashboardWindow       : main hub with stats + navigation buttons
    - RegisterStudentWindow : add a new student + capture + train
    - AttendanceWindow      : start real-time recognition
    - ViewAttendanceWindow  : browse / search / export attendance
    - ManageStudentsWindow  : update / delete students

The GUI layer only handles presentation and input validation feedback;
all business logic lives in database.py, face_detector.py,
face_trainer.py and attendance.py.
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import pandas as pd

from database import Database, DatabaseError
from face_detector import FaceDetector, FaceCaptureError
from face_trainer import FaceTrainer, FaceTrainerError
from attendance import AttendanceSystem, AttendanceError
from utils import (
    validate_name,
    validate_roll_number,
    validate_department,
    validate_email,
    get_today_date,
    ATTENDANCE_DIR,
    logger,
)

# ---------------------------------------------------------------------------
# Shared style constants - keeps the "modern, clean" look consistent
# ---------------------------------------------------------------------------
BG_COLOR = "#1e293b"        # slate 800
CARD_COLOR = "#f8fafc"      # slate 50
ACCENT_COLOR = "#2563eb"    # blue 600
ACCENT_DARK = "#1e40af"
SUCCESS_COLOR = "#16a34a"
DANGER_COLOR = "#dc2626"
TEXT_DARK = "#0f172a"
FONT_HEADING = ("Segoe UI", 20, "bold")
FONT_SUBHEADING = ("Segoe UI", 12, "bold")
FONT_BODY = ("Segoe UI", 10)


def styled_button(parent, text, command, bg=ACCENT_COLOR, fg="white", width=20):
    btn = tk.Button(
        parent, text=text, command=command, bg=bg, fg=fg,
        font=FONT_SUBHEADING, relief="flat", width=width, height=2,
        activebackground=ACCENT_DARK, activeforeground="white", cursor="hand2",
    )
    return btn


class LoginScreen(tk.Frame):
    """A lightweight login gate. Credentials are intentionally simple since
    this project's focus is face recognition, not user-auth security."""

    VALID_USERNAME = "admin"
    VALID_PASSWORD = "admin123"

    def __init__(self, master, on_success):
        super().__init__(master, bg=BG_COLOR)
        self.on_success = on_success
        self.pack(fill="both", expand=True)
        self._build_ui()

    def _build_ui(self):
        card = tk.Frame(self, bg=CARD_COLOR, padx=40, pady=40)
        card.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(
            card, text="Face Recognition\nAttendance System", font=FONT_HEADING,
            bg=CARD_COLOR, fg=TEXT_DARK, justify="center",
        ).pack(pady=(0, 20))

        tk.Label(card, text="Username", font=FONT_BODY, bg=CARD_COLOR).pack(anchor="w")
        self.username_entry = tk.Entry(card, font=FONT_BODY, width=30)
        self.username_entry.pack(pady=(0, 10))
        self.username_entry.insert(0, "admin")

        tk.Label(card, text="Password", font=FONT_BODY, bg=CARD_COLOR).pack(anchor="w")
        self.password_entry = tk.Entry(card, font=FONT_BODY, width=30, show="*")
        self.password_entry.pack(pady=(0, 20))
        self.password_entry.insert(0, "admin123")

        styled_button(card, "Login", self._attempt_login).pack(fill="x")
        tk.Label(
            card, text="Default: admin / admin123", font=("Segoe UI", 8),
            bg=CARD_COLOR, fg="#64748b",
        ).pack(pady=(10, 0))

        self.password_entry.bind("<Return>", lambda e: self._attempt_login())

    def _attempt_login(self):
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()
        if username == self.VALID_USERNAME and password == self.VALID_PASSWORD:
            self.on_success()
        else:
            messagebox.showerror("Login Failed", "Invalid username or password.")


class DashboardWindow(tk.Frame):
    """Main hub shown after login; displays live stats and navigation."""

    def __init__(self, master, db: Database):
        super().__init__(master, bg=BG_COLOR)
        self.db = db
        self.pack(fill="both", expand=True)
        self._build_ui()
        self.refresh_stats()

    def _build_ui(self):
        header = tk.Frame(self, bg=BG_COLOR, pady=20)
        header.pack(fill="x")
        tk.Label(
            header, text="📊 Attendance Dashboard", font=FONT_HEADING,
            bg=BG_COLOR, fg="white",
        ).pack(side="left", padx=30)
        tk.Label(
            header, text=f"Date: {get_today_date()}", font=FONT_BODY,
            bg=BG_COLOR, fg="#cbd5e1",
        ).pack(side="right", padx=30)

        # ---- Stat cards -------------------------------------------------
        stats_frame = tk.Frame(self, bg=BG_COLOR)
        stats_frame.pack(fill="x", padx=30, pady=10)

        self.stat_labels = {}
        stat_defs = [
            ("total_students", "Total Students", ACCENT_COLOR),
            ("present_today", "Present Today", SUCCESS_COLOR),
            ("absent_today", "Absent Today", DANGER_COLOR),
        ]
        for key, title, color in stat_defs:
            card = tk.Frame(stats_frame, bg=CARD_COLOR, padx=20, pady=20)
            card.pack(side="left", expand=True, fill="both", padx=10)
            value_label = tk.Label(
                card, text="0", font=("Segoe UI", 28, "bold"), bg=CARD_COLOR, fg=color
            )
            value_label.pack()
            tk.Label(card, text=title, font=FONT_BODY, bg=CARD_COLOR, fg=TEXT_DARK).pack()
            self.stat_labels[key] = value_label

        # ---- Navigation buttons ------------------------------------------
        nav_frame = tk.Frame(self, bg=BG_COLOR, pady=20)
        nav_frame.pack(fill="both", expand=True, padx=30)

        buttons = [
            ("➕ Register Student", self.open_register, ACCENT_COLOR),
            ("📷 Take Attendance", self.open_attendance, SUCCESS_COLOR),
            ("📋 View Attendance", self.open_view_attendance, ACCENT_COLOR),
            ("🛠 Manage Students", self.open_manage_students, "#7c3aed"),
            ("🔄 Refresh Stats", self.refresh_stats, "#475569"),
        ]
        for idx, (text, cmd, color) in enumerate(buttons):
            b = styled_button(nav_frame, text, cmd, bg=color, width=25)
            b.grid(row=idx // 2, column=idx % 2, padx=15, pady=15, sticky="nsew")
        nav_frame.grid_columnconfigure(0, weight=1)
        nav_frame.grid_columnconfigure(1, weight=1)

    def refresh_stats(self):
        try:
            stats = self.db.get_dashboard_stats()
            self.stat_labels["total_students"].config(text=str(stats["total_students"]))
            self.stat_labels["present_today"].config(text=str(stats["present_today"]))
            self.stat_labels["absent_today"].config(text=str(stats["absent_today"]))
        except DatabaseError as exc:
            messagebox.showerror("Database Error", str(exc))

    def open_register(self):
        RegisterStudentWindow(tk.Toplevel(self), self.db, on_close=self.refresh_stats)

    def open_attendance(self):
        AttendanceWindow(tk.Toplevel(self), self.db, on_close=self.refresh_stats)

    def open_view_attendance(self):
        ViewAttendanceWindow(tk.Toplevel(self), self.db)

    def open_manage_students(self):
        ManageStudentsWindow(tk.Toplevel(self), self.db, on_close=self.refresh_stats)


class RegisterStudentWindow:
    """Form to register a new student, capture face images, and retrain."""

    def __init__(self, window: tk.Toplevel, db: Database, on_close=None):
        self.window = window
        self.db = db
        self.on_close = on_close
        window.title("Register New Student")
        window.geometry("450x520")
        window.configure(bg=CARD_COLOR)
        self._build_ui()

    def _build_ui(self):
        tk.Label(
            self.window, text="Register New Student", font=FONT_HEADING,
            bg=CARD_COLOR, fg=TEXT_DARK,
        ).pack(pady=20)

        form = tk.Frame(self.window, bg=CARD_COLOR)
        form.pack(padx=30, fill="x")

        self.entries = {}
        fields = [("Name", "name"), ("Roll Number", "roll"),
                  ("Department", "dept"), ("Email", "email")]
        for label_text, key in fields:
            tk.Label(form, text=label_text, font=FONT_BODY, bg=CARD_COLOR).pack(anchor="w")
            entry = tk.Entry(form, font=FONT_BODY, width=35)
            entry.pack(pady=(0, 12))
            self.entries[key] = entry

        self.status_label = tk.Label(
            self.window, text="", font=FONT_BODY, bg=CARD_COLOR, fg=ACCENT_COLOR, wraplength=380
        )
        self.status_label.pack(pady=10)

        btn_frame = tk.Frame(self.window, bg=CARD_COLOR)
        btn_frame.pack(pady=10)
        styled_button(
            btn_frame, "Capture Face & Register", self._register_and_capture,
            width=28,
        ).pack(pady=5)
        styled_button(
            btn_frame, "Train Model", self._train_model, bg="#7c3aed", width=28
        ).pack(pady=5)

    def _validate_inputs(self):
        name = self.entries["name"].get().strip()
        roll = self.entries["roll"].get().strip()
        dept = self.entries["dept"].get().strip()
        email = self.entries["email"].get().strip()

        if not validate_name(name):
            raise ValueError("Name must contain only letters/spaces (2-50 chars).")
        if not validate_roll_number(roll):
            raise ValueError("Roll number must be alphanumeric (2-20 chars).")
        if not validate_department(dept):
            raise ValueError("Department must be at least 2 characters.")
        if not validate_email(email):
            raise ValueError("Please enter a valid email address.")

        return name, roll, dept, email

    def _register_and_capture(self):
        try:
            name, roll, dept, email = self._validate_inputs()

            if self.db.get_student_by_roll(roll):
                raise ValueError(f"Roll number '{roll}' is already registered.")

            student_id = self.db.add_student(name, roll, dept, email)
            self.status_label.config(
                text="Student saved. Opening webcam to capture face images...\n"
                     "Look at the camera and press Q to cancel early.",
                fg=ACCENT_COLOR,
            )
            self.window.update_idletasks()

            detector = FaceDetector(num_images=40)

            def do_capture():
                try:
                    detector.capture_faces(roll, name)
                    self.window.after(0, lambda: self.status_label.config(
                        text=f"Face images captured for {name}. Click 'Train Model' next.",
                        fg=SUCCESS_COLOR,
                    ))
                except FaceCaptureError as exc:
                    self.window.after(0, lambda: messagebox.showerror("Capture Error", str(exc)))

            threading.Thread(target=do_capture, daemon=True).start()

        except (ValueError, DatabaseError) as exc:
            messagebox.showerror("Validation Error", str(exc))

    def _train_model(self):
        self.status_label.config(text="Training model, please wait...", fg=ACCENT_COLOR)
        self.window.update_idletasks()

        def do_train():
            try:
                trainer = FaceTrainer()
                num_encodings, num_students = trainer.train()
                msg = f"Training complete: {num_encodings} encodings for {num_students} students."
                self.window.after(0, lambda: self.status_label.config(text=msg, fg=SUCCESS_COLOR))
                self.window.after(0, lambda: messagebox.showinfo("Training Complete", msg))
                if self.on_close:
                    self.window.after(0, self.on_close)
            except FaceTrainerError as exc:
                self.window.after(0, lambda: messagebox.showerror("Training Error", str(exc)))

        threading.Thread(target=do_train, daemon=True).start()


class AttendanceWindow:
    """Launches real-time face recognition to mark attendance automatically."""

    def __init__(self, window: tk.Toplevel, db: Database, on_close=None):
        self.window = window
        self.db = db
        self.on_close = on_close
        self._stop = False
        window.title("Take Attendance")
        window.geometry("450x350")
        window.configure(bg=CARD_COLOR)
        self._build_ui()

    def _build_ui(self):
        tk.Label(
            self.window, text="📷 Real-Time Attendance", font=FONT_HEADING,
            bg=CARD_COLOR, fg=TEXT_DARK,
        ).pack(pady=20)
        tk.Label(
            self.window,
            text="Click Start, then look at the camera.\nPress Q in the camera window (or click Stop) to finish.",
            font=FONT_BODY, bg=CARD_COLOR, justify="center",
        ).pack(pady=10)

        self.log_box = tk.Listbox(self.window, width=50, height=8, font=("Consolas", 9))
        self.log_box.pack(pady=10, padx=20, fill="both", expand=True)

        btn_frame = tk.Frame(self.window, bg=CARD_COLOR)
        btn_frame.pack(pady=10)
        styled_button(btn_frame, "Start Recognition", self._start, bg=SUCCESS_COLOR, width=18).pack(
            side="left", padx=5
        )
        styled_button(btn_frame, "Stop", self._stop_recognition, bg=DANGER_COLOR, width=18).pack(
            side="left", padx=5
        )

    def _log(self, message):
        self.log_box.insert("end", message)
        self.log_box.see("end")

    def _start(self):
        self._stop = False

        def run():
            try:
                system = AttendanceSystem(self.db)
                system.run_recognition(
                    status_callback=lambda m: self.window.after(0, self._log, m),
                    stop_flag=lambda: self._stop,
                )
            except AttendanceError as exc:
                self.window.after(0, lambda: messagebox.showerror("Attendance Error", str(exc)))
            finally:
                if self.on_close:
                    self.window.after(0, self.on_close)

        threading.Thread(target=run, daemon=True).start()

    def _stop_recognition(self):
        self._stop = True


class ViewAttendanceWindow:
    """Browse, search, and export attendance records."""

    def __init__(self, window: tk.Toplevel, db: Database):
        self.window = window
        self.db = db
        window.title("View Attendance")
        window.geometry("850x500")
        window.configure(bg=CARD_COLOR)
        self._build_ui()
        self._load_all()

    def _build_ui(self):
        tk.Label(
            self.window, text="📋 Attendance Records", font=FONT_HEADING,
            bg=CARD_COLOR, fg=TEXT_DARK,
        ).pack(pady=15)

        toolbar = tk.Frame(self.window, bg=CARD_COLOR)
        toolbar.pack(fill="x", padx=20)

        self.search_entry = tk.Entry(toolbar, font=FONT_BODY, width=30)
        self.search_entry.pack(side="left", padx=(0, 10))
        styled_button(toolbar, "Search", self._search, width=10).pack(side="left", padx=5)
        styled_button(toolbar, "Show All", self._load_all, bg="#475569", width=10).pack(
            side="left", padx=5
        )
        styled_button(toolbar, "Today Only", self._load_today, bg=ACCENT_COLOR, width=10).pack(
            side="left", padx=5
        )
        styled_button(toolbar, "Export to CSV", self._export_csv, bg=SUCCESS_COLOR, width=14).pack(
            side="right"
        )

        columns = ("id", "name", "roll", "dept", "date", "time", "status")
        self.tree = ttk.Treeview(self.window, columns=columns, show="headings", height=18)
        headings = ["ID", "Name", "Roll No.", "Department", "Date", "Time", "Status"]
        for col, heading in zip(columns, headings):
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=110, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=20, pady=15)

    def _populate(self, rows):
        self.tree.delete(*self.tree.get_children())
        for row in rows:
            self.tree.insert("", "end", values=row)

    def _load_all(self):
        try:
            self._populate(self.db.get_all_attendance())
        except DatabaseError as exc:
            messagebox.showerror("Database Error", str(exc))

    def _load_today(self):
        try:
            self._populate(self.db.get_attendance_by_date())
        except DatabaseError as exc:
            messagebox.showerror("Database Error", str(exc))

    def _search(self):
        keyword = self.search_entry.get().strip()
        if not keyword:
            self._load_all()
            return
        try:
            self._populate(self.db.search_attendance(keyword))
        except DatabaseError as exc:
            messagebox.showerror("Database Error", str(exc))

    def _export_csv(self):
        rows = [self.tree.item(item)["values"] for item in self.tree.get_children()]
        if not rows:
            messagebox.showwarning("No Data", "There are no records to export.")
            return

        columns = ["Attendance ID", "Name", "Roll No.", "Department", "Date", "Time", "Status"]
        df = pd.DataFrame(rows, columns=columns)

        default_name = f"attendance_export_{get_today_date()}.csv"
        file_path = filedialog.asksaveasfilename(
            initialdir=ATTENDANCE_DIR,
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not file_path:
            return
        try:
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Export Successful", f"Attendance exported to:\n{file_path}")
        except OSError as exc:
            messagebox.showerror("Export Error", f"Could not save file: {exc}")


class ManageStudentsWindow:
    """List, update, and delete student records."""

    def __init__(self, window: tk.Toplevel, db: Database, on_close=None):
        self.window = window
        self.db = db
        self.on_close = on_close
        window.title("Manage Students")
        window.geometry("750x480")
        window.configure(bg=CARD_COLOR)
        self._build_ui()
        self._load_students()

    def _build_ui(self):
        tk.Label(
            self.window, text="🛠 Manage Students", font=FONT_HEADING,
            bg=CARD_COLOR, fg=TEXT_DARK,
        ).pack(pady=15)

        columns = ("id", "name", "roll", "dept", "email")
        self.tree = ttk.Treeview(self.window, columns=columns, show="headings", height=14)
        for col, heading in zip(columns, ["ID", "Name", "Roll No.", "Department", "Email"]):
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=130, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=20, pady=10)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        form = tk.Frame(self.window, bg=CARD_COLOR)
        form.pack(fill="x", padx=20)

        self.name_entry = self._labeled_entry(form, "Name", 0)
        self.dept_entry = self._labeled_entry(form, "Department", 1)
        self.email_entry = self._labeled_entry(form, "Email", 2)

        btn_frame = tk.Frame(self.window, bg=CARD_COLOR)
        btn_frame.pack(pady=10)
        styled_button(btn_frame, "Update Selected", self._update_student, width=18).pack(
            side="left", padx=5
        )
        styled_button(btn_frame, "Delete Selected", self._delete_student, bg=DANGER_COLOR, width=18).pack(
            side="left", padx=5
        )

        self.selected_id = None

    def _labeled_entry(self, parent, label, col):
        tk.Label(parent, text=label, font=FONT_BODY, bg=CARD_COLOR).grid(
            row=0, column=col, padx=5, sticky="w"
        )
        entry = tk.Entry(parent, font=FONT_BODY, width=20)
        entry.grid(row=1, column=col, padx=5, pady=5)
        return entry

    def _load_students(self):
        try:
            self.tree.delete(*self.tree.get_children())
            for row in self.db.get_all_students():
                self.tree.insert("", "end", values=row)
        except DatabaseError as exc:
            messagebox.showerror("Database Error", str(exc))

    def _on_select(self, event):
        selection = self.tree.selection()
        if not selection:
            return
        values = self.tree.item(selection[0])["values"]
        self.selected_id = values[0]
        self.name_entry.delete(0, "end")
        self.name_entry.insert(0, values[1])
        self.dept_entry.delete(0, "end")
        self.dept_entry.insert(0, values[3])
        self.email_entry.delete(0, "end")
        self.email_entry.insert(0, values[4])

    def _update_student(self):
        if not self.selected_id:
            messagebox.showwarning("No Selection", "Please select a student from the list.")
            return
        name = self.name_entry.get().strip()
        dept = self.dept_entry.get().strip()
        email = self.email_entry.get().strip()

        try:
            if not validate_name(name):
                raise ValueError("Name must contain only letters/spaces (2-50 chars).")
            if not validate_department(dept):
                raise ValueError("Department must be at least 2 characters.")
            if not validate_email(email):
                raise ValueError("Please enter a valid email address.")

            self.db.update_student(self.selected_id, name, dept, email)
            messagebox.showinfo("Success", "Student updated successfully.")
            self._load_students()
            if self.on_close:
                self.on_close()
        except (ValueError, DatabaseError) as exc:
            messagebox.showerror("Error", str(exc))

    def _delete_student(self):
        if not self.selected_id:
            messagebox.showwarning("No Selection", "Please select a student from the list.")
            return
        confirm = messagebox.askyesno(
            "Confirm Delete", "This will permanently delete the student and their attendance history. Continue?"
        )
        if not confirm:
            return
        try:
            self.db.delete_student(self.selected_id)
            messagebox.showinfo("Deleted", "Student deleted successfully.")
            self.selected_id = None
            self._load_students()
            if self.on_close:
                self.on_close()
        except DatabaseError as exc:
            messagebox.showerror("Error", str(exc))


class App:
    """Top-level application controller that swaps frames (Login -> Dashboard)."""

    def __init__(self, root: tk.Tk, db: Database):
        self.root = root
        self.db = db
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1000x650")
        self.root.configure(bg=BG_COLOR)
        self._show_login()

    def _show_login(self):
        self._clear_root()
        LoginScreen(self.root, on_success=self._show_dashboard)

    def _show_dashboard(self):
        self._clear_root()
        DashboardWindow(self.root, self.db)

    def _clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()
