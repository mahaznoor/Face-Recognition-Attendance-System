"""
utils.py
--------
Common helper functions used across the Face Recognition Attendance System:
input validation, date/time helpers, directory setup and simple logging.

Keeping these utilities in one module avoids code duplication and keeps
the rest of the codebase focused on its own responsibility (SRP).
"""

import os
import re
import logging
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths (relative to project root, since main.py is expected to be run
# from the project root or src/ folder - both are handled below).
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINED_MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

DATABASE_PATH = os.path.join(DATABASE_DIR, "attendance.db")
ENCODINGS_PATH = os.path.join(TRAINED_MODEL_DIR, "encodings.pickle")


def ensure_directories():
    """Create all required project directories if they do not already exist."""
    for directory in (
        DATABASE_DIR,
        DATASET_DIR,
        TRAINED_MODEL_DIR,
        ATTENDANCE_DIR,
        SCREENSHOTS_DIR,
    ):
        os.makedirs(directory, exist_ok=True)


def setup_logger(name="FaceAttendanceSystem"):
    """Configure and return a simple console + file logger."""
    ensure_directories()
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(BASE_DIR, "app.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
NAME_REGEX = re.compile(r"^[A-Za-z ]{2,50}$")
ROLL_REGEX = re.compile(r"^[A-Za-z0-9\-/]{2,20}$")
EMAIL_REGEX = re.compile(r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$")


def validate_name(name: str) -> bool:
    """Name should contain only letters and spaces, 2-50 characters."""
    return bool(name) and bool(NAME_REGEX.match(name.strip()))


def validate_roll_number(roll_number: str) -> bool:
    """Roll number is alphanumeric (dashes/slashes allowed), 2-20 characters."""
    return bool(roll_number) and bool(ROLL_REGEX.match(roll_number.strip()))


def validate_department(department: str) -> bool:
    """Department must simply be a non-empty string."""
    return bool(department) and len(department.strip()) >= 2


def validate_email(email: str) -> bool:
    """Basic email format validation."""
    return bool(email) and bool(EMAIL_REGEX.match(email.strip()))


# ---------------------------------------------------------------------------
# Date / time helpers
# ---------------------------------------------------------------------------
def get_today_date() -> str:
    """Return today's date as YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")


def get_current_time() -> str:
    """Return the current time as HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")


def get_timestamp() -> str:
    """Return a filesystem-safe timestamp, useful for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_folder_name(roll_number: str, name: str) -> str:
    """Build a safe, unique folder name for a student's dataset images."""
    clean_name = re.sub(r"[^A-Za-z0-9]", "_", name.strip())
    clean_roll = re.sub(r"[^A-Za-z0-9]", "_", roll_number.strip())
    return f"{clean_roll}_{clean_name}"
