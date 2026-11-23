"""
face_detector.py
-----------------
Responsible for capturing face images for a newly registered student
using the webcam and OpenCV's Haar Cascade face detector.

The captured, cropped face images are stored under:
    dataset/<roll_number>_<name>/img_0.jpg, img_1.jpg, ...

These images are later consumed by face_trainer.py to generate
face embeddings using DeepFace.
"""

import os
import cv2

from utils import DATASET_DIR, safe_folder_name, logger, ensure_directories


class FaceCaptureError(Exception):
    """Raised when the webcam or capture process fails."""
    pass


class FaceDetector:
    """Captures and stores face images for a student using the webcam."""

    def __init__(self, num_images: int = 50):
        ensure_directories()
        self.num_images = num_images
        # Haar cascade shipped with OpenCV - no extra download required.
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise FaceCaptureError("Could not load Haar Cascade classifier.")

    def capture_faces(self, roll_number: str, name: str, progress_callback=None):
        """
        Open the webcam, detect faces, and save cropped grayscale-free
        face crops to disk. Returns the folder path where images were saved.

        progress_callback(count, total) is optionally invoked after every
        saved image so the GUI can update a progress bar/label.
        """
        folder_name = safe_folder_name(roll_number, name)
        student_dir = os.path.join(DATASET_DIR, folder_name)
        os.makedirs(student_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise FaceCaptureError("Could not access the webcam. Check camera connection.")

        count = 0
        try:
            while count < self.num_images:
                ret, frame = cap.read()
                if not ret:
                    raise FaceCaptureError("Failed to read frame from webcam.")

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
                )

                for (x, y, w, h) in faces:
                    count += 1
                    face_crop = frame[y:y + h, x:x + w]
                    img_path = os.path.join(student_dir, f"img_{count}.jpg")
                    cv2.imwrite(img_path, face_crop)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"Captured {count}/{self.num_images}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )
                    if progress_callback:
                        progress_callback(count, self.num_images)
                    break  # only take one face per frame

                cv2.imshow("Registering Face - Press Q to cancel", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if count >= self.num_images:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        if count == 0:
            # Clean up empty folder to avoid confusing the trainer later.
            try:
                os.rmdir(student_dir)
            except OSError:
                pass
            raise FaceCaptureError("No face detected. Please try again with better lighting.")

        logger.info(f"Captured {count} face images for {name} ({roll_number})")
        return student_dir
