"""
database.py
------------
Handles all SQLite database operations for the Face Recognition
Attendance System using a single Database class (OOP design).

Tables
------
students:
    id            INTEGER PRIMARY KEY AUTOINCREMENT
    name          TEXT NOT NULL
    roll_number   TEXT UNIQUE NOT NULL
    department    TEXT NOT NULL
    email         TEXT NOT NULL

attendance:
    attendance_id INTEGER PRIMARY KEY AUTOINCREMENT
    student_id    INTEGER NOT NULL (FK -> students.id)
    date          TEXT NOT NULL
    time          TEXT NOT NULL
    status        TEXT NOT NULL   -- 'Present' / 'Absent'
"""

import sqlite3
from sqlite3 import Error as SQLiteError

from utils import DATABASE_PATH, ensure_directories, logger, get_today_date


class DatabaseError(Exception):
    """Custom exception wrapping low level sqlite3 errors."""
    pass


class Database:
    """Encapsulates every SQL operation needed by the application."""

    def __init__(self, db_path: str = DATABASE_PATH):
        ensure_directories()
        self.db_path = db_path
        self._create_tables()

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------
    def _get_connection(self):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            return conn
        except SQLiteError as exc:
            logger.error(f"Could not connect to database: {exc}")
            raise DatabaseError(f"Could not connect to database: {exc}") from exc

    def _create_tables(self):
        """Create the students and attendance tables if they don't exist."""
        create_students_sql = """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                roll_number TEXT UNIQUE NOT NULL,
                department TEXT NOT NULL,
                email TEXT NOT NULL
            );
        """
        create_attendance_sql = """
            CREATE TABLE IF NOT EXISTS attendance (
                attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students (id)
                    ON DELETE CASCADE
            );
        """
        try:
            with self._get_connection() as conn:
                conn.execute(create_students_sql)
                conn.execute(create_attendance_sql)
                conn.commit()
        except SQLiteError as exc:
            logger.error(f"Error creating tables: {exc}")
            raise DatabaseError(f"Error creating tables: {exc}") from exc

    # ------------------------------------------------------------------
    # Student CRUD
    # ------------------------------------------------------------------
    def add_student(self, name, roll_number, department, email) -> int:
        """Insert a new student and return the generated student id."""
        sql = """INSERT INTO students (name, roll_number, department, email)
                 VALUES (?, ?, ?, ?)"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(sql, (name, roll_number, department, email))
                conn.commit()
                return cursor.lastrowid
        except sqlite3.IntegrityError as exc:
            raise DatabaseError(f"Roll number '{roll_number}' already exists.") from exc
        except SQLiteError as exc:
            raise DatabaseError(f"Error adding student: {exc}") from exc

    def get_all_students(self):
        """Return a list of tuples: (id, name, roll_number, department, email)."""
        sql = "SELECT id, name, roll_number, department, email FROM students ORDER BY id"
        try:
            with self._get_connection() as conn:
                return conn.execute(sql).fetchall()
        except SQLiteError as exc:
            raise DatabaseError(f"Error fetching students: {exc}") from exc

    def get_student_by_roll(self, roll_number):
        sql = "SELECT id, name, roll_number, department, email FROM students WHERE roll_number = ?"
        try:
            with self._get_connection() as conn:
                return conn.execute(sql, (roll_number,)).fetchone()
        except SQLiteError as exc:
            raise DatabaseError(f"Error fetching student: {exc}") from exc

    def get_student_by_id(self, student_id):
        sql = "SELECT id, name, roll_number, department, email FROM students WHERE id = ?"
        try:
            with self._get_connection() as conn:
                return conn.execute(sql, (student_id,)).fetchone()
        except SQLiteError as exc:
            raise DatabaseError(f"Error fetching student: {exc}") from exc

    def update_student(self, student_id, name, department, email):
        sql = "UPDATE students SET name = ?, department = ?, email = ? WHERE id = ?"
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(sql, (name, department, email, student_id))
                conn.commit()
                if cursor.rowcount == 0:
                    raise DatabaseError("No student found with that ID.")
        except SQLiteError as exc:
            raise DatabaseError(f"Error updating student: {exc}") from exc

    def delete_student(self, student_id):
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
                cursor = conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
                conn.commit()
                if cursor.rowcount == 0:
                    raise DatabaseError("No student found with that ID.")
        except SQLiteError as exc:
            raise DatabaseError(f"Error deleting student: {exc}") from exc

    # ------------------------------------------------------------------
    # Attendance
    # ------------------------------------------------------------------
    def has_marked_today(self, student_id, date=None) -> bool:
        """Prevents duplicate attendance for the same student on the same day."""
        date = date or get_today_date()
        sql = "SELECT 1 FROM attendance WHERE student_id = ? AND date = ? LIMIT 1"
        try:
            with self._get_connection() as conn:
                return conn.execute(sql, (student_id, date)).fetchone() is not None
        except SQLiteError as exc:
            raise DatabaseError(f"Error checking attendance: {exc}") from exc

    def mark_attendance(self, student_id, date, time_, status="Present"):
        """Insert an attendance record if not already marked for the given date."""
        if self.has_marked_today(student_id, date):
            return False  # duplicate prevented
        sql = "INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, ?)"
        try:
            with self._get_connection() as conn:
                conn.execute(sql, (student_id, date, time_, status))
                conn.commit()
                return True
        except SQLiteError as exc:
            raise DatabaseError(f"Error marking attendance: {exc}") from exc

    def get_attendance_by_date(self, date=None):
        """Return joined attendance + student info for a given date."""
        date = date or get_today_date()
        sql = """
            SELECT a.attendance_id, s.name, s.roll_number, s.department,
                   a.date, a.time, a.status
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.date = ?
            ORDER BY a.time
        """
        try:
            with self._get_connection() as conn:
                return conn.execute(sql, (date,)).fetchall()
        except SQLiteError as exc:
            raise DatabaseError(f"Error fetching attendance: {exc}") from exc

    def get_all_attendance(self):
        sql = """
            SELECT a.attendance_id, s.name, s.roll_number, s.department,
                   a.date, a.time, a.status
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            ORDER BY a.date DESC, a.time DESC
        """
        try:
            with self._get_connection() as conn:
                return conn.execute(sql).fetchall()
        except SQLiteError as exc:
            raise DatabaseError(f"Error fetching attendance: {exc}") from exc

    def search_attendance(self, keyword):
        """Search attendance records by student name or roll number."""
        sql = """
            SELECT a.attendance_id, s.name, s.roll_number, s.department,
                   a.date, a.time, a.status
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE s.name LIKE ? OR s.roll_number LIKE ?
            ORDER BY a.date DESC, a.time DESC
        """
        like_kw = f"%{keyword}%"
        try:
            with self._get_connection() as conn:
                return conn.execute(sql, (like_kw, like_kw)).fetchall()
        except SQLiteError as exc:
            raise DatabaseError(f"Error searching attendance: {exc}") from exc

    # ------------------------------------------------------------------
    # Dashboard statistics
    # ------------------------------------------------------------------
    def get_dashboard_stats(self):
        """Return dict with total students, present today, absent today."""
        today = get_today_date()
        try:
            with self._get_connection() as conn:
                total_students = conn.execute(
                    "SELECT COUNT(*) FROM students"
                ).fetchone()[0]
                present_today = conn.execute(
                    "SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date = ?",
                    (today,),
                ).fetchone()[0]
                absent_today = max(total_students - present_today, 0)
                return {
                    "total_students": total_students,
                    "present_today": present_today,
                    "absent_today": absent_today,
                    "date": today,
                }
        except SQLiteError as exc:
            raise DatabaseError(f"Error fetching dashboard stats: {exc}") from exc
