"""
attendance.py
-------------
Handles real-time face recognition using the webcam and marks
attendance automatically in the database once a known face is matched
with sufficient confidence.

Face *detection* (finding where a face is in the frame) is done with
OpenCV's Haar Cascade, exactly like face_detector.py during
registration. Face *recognition* (deciding whose face it is) is done by
computing a DeepFace embedding for the detected face crop and comparing
it against the known embeddings saved by face_trainer.py using cosine
distance. No dlib / face_recognition dependency is required.

Duplicate attendance for the same student on the same day is prevented
at the database layer (Database.has_marked_today / mark_attendance).
"""

import os
import pickle

import cv2
import numpy as np
from deepface import DeepFace

from utils import (
    ENCODINGS_PATH,
    SCREENSHOTS_DIR,
    get_today_date,
    get_current_time,
    get_timestamp,
    logger,
    ensure_directories,
)
from database import Database, DatabaseError
from face_trainer import EMBEDDING_MODEL, DETECTOR_BACKEND


class AttendanceError(Exception):
    """Raised for webcam or model-loading failures."""
    pass


class AttendanceSystem:
    """Runs real-time face recognition and records attendance."""

    # Cosine distance: 0 = identical, 2 = opposite. DeepFace's own default
    # verification threshold for Facenet + cosine distance is ~0.40; we use
    # the same value here. Lower it for stricter matching, raise it if
    # correct matches are being rejected under poor lighting.
    MATCH_THRESHOLD = 0.40

    def __init__(self, db: Database):
        ensure_directories()
        self.db = db
        self.known_encodings = None      # np.ndarray, shape (N, D), L2-normalized
        self.known_roll_numbers = []
        self._load_encodings()

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise AttendanceError("Could not load Haar Cascade classifier.")

    def _load_encodings(self):
        if not os.path.exists(ENCODINGS_PATH):
            raise AttendanceError("No trained model found. Please train the model first.")
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)

        encodings = data.get("encodings", [])
        roll_numbers = data.get("roll_numbers", [])
        model_name = data.get("model_name")

        if not encodings:
            raise AttendanceError("Trained model is empty. Please retrain.")
        if model_name and model_name != EMBEDDING_MODEL:
            raise AttendanceError(
                f"Trained model uses '{model_name}' but the app is configured for "
                f"'{EMBEDDING_MODEL}'. Please retrain the model."
            )

        self.known_encodings = np.vstack(encodings).astype(np.float32)
        self.known_roll_numbers = roll_numbers

    def run_recognition(self, status_callback=None, stop_flag=None):
        """
        Open the webcam and continuously recognize faces until the user
        presses 'q' or stop_flag() returns True (used by the GUI to allow
        stopping the loop from a button click).

        status_callback(message) is called with human-readable status
        updates (e.g. "Marked present: John Doe (CS101)").
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise AttendanceError("Could not access the webcam.")

        marked_this_session = set()

        try:
            while True:
                if stop_flag and stop_flag():
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
                )

                for (x, y, w, h) in faces:
                    face_crop = frame[y:y + h, x:x + w]
                    roll_number, name_label = self._match_face(face_crop)

                    color = (0, 200, 0) if roll_number else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        frame, name_label, (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    )

                    if roll_number and roll_number not in marked_this_session:
                        marked_this_session.add(roll_number)
                        self._mark_and_notify(roll_number, frame, status_callback)

                cv2.imshow("Attendance - Press Q to stop", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _match_face(self, face_crop):
        """Compare a live face crop against known encodings. Returns (roll_number, label)."""
        if self.known_encodings is None or len(self.known_encodings) == 0:
            return None, "Unknown"

        embedding = self._get_embedding(face_crop)
        if embedding is None:
            return None, "Unknown"

        # Cosine distance = 1 - cosine similarity. Both sides are L2-normalized,
        # so a plain dot product gives the cosine similarity directly.
        similarities = self.known_encodings @ embedding
        distances = 1.0 - similarities
        best_index = int(np.argmin(distances))

        if distances[best_index] <= self.MATCH_THRESHOLD:
            roll_number = self.known_roll_numbers[best_index]
            student = self.db.get_student_by_roll(roll_number)
            label = student[1] if student else roll_number
            return roll_number, label
        return None, "Unknown"

    @staticmethod
    def _get_embedding(face_crop):
        """Return a single L2-normalized embedding (np.ndarray) for a face crop, or None."""
        try:
            reps = DeepFace.represent(
                img_path=face_crop,
                model_name=EMBEDDING_MODEL,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"DeepFace could not embed live frame: {exc}")
            return None

        if not reps:
            return None

        embedding = np.array(reps[0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None
        return embedding / norm

    def _mark_and_notify(self, roll_number, frame, status_callback):
        try:
            student = self.db.get_student_by_roll(roll_number)
            if not student:
                return
            student_id, name = student[0], student[1]

            marked = self.db.mark_attendance(
                student_id, get_today_date(), get_current_time(), status="Present"
            )
            if marked:
                screenshot_path = os.path.join(
                    SCREENSHOTS_DIR, f"{roll_number}_{get_timestamp()}.jpg"
                )
                cv2.imwrite(screenshot_path, frame)
                message = f"Marked present: {name} ({roll_number})"
                logger.info(message)
            else:
                message = f"{name} ({roll_number}) already marked present today."

            if status_callback:
                status_callback(message)
        except DatabaseError as exc:
            logger.error(f"Error marking attendance: {exc}")
            if status_callback:
                status_callback(f"Error marking attendance: {exc}")
