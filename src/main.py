"""
main.py
-------
Entry point for the Face Recognition Attendance System.

Run this file to launch the application:
    python src/main.py

It initializes the SQLite database (creating tables if needed),
ensures all required project folders exist, and then starts the
Tkinter GUI.
"""

import sys
import tkinter as tk

from utils import ensure_directories, logger
from database import Database, DatabaseError
import tensorflow.keras.preprocessing
from gui import App


def main():
    ensure_directories()
    try:
        db = Database()
    except DatabaseError as exc:
        logger.error(f"Fatal database error: {exc}")
        print(f"Fatal database error: {exc}")
        sys.exit(1)

    root = tk.Tk()
    App(root, db)
    root.mainloop()


if __name__ == "__main__":
    main()
