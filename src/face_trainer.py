"""
face_trainer.py
---------------
Trains (encodes) the face recognition model.

"Training" here means: walk through every student folder inside dataset/,
compute a face embedding for each cropped face image using DeepFace
(no dlib / face_recognition dependency), and persist the embeddings +
labels (roll numbers) to trained_model/encodings.pickle.

DeepFace.represent() is used purely as an *embedding extractor*. Since
the images in dataset/ are already tightly cropped face images produced
by face_detector.py (OpenCV Haar Cascade), we tell DeepFace to skip its
own internal face detector (detector_backend="skip") for speed and to
avoid discarding valid crops that a stricter detector might miss.

attendance.py loads this pickle file at runtime to compare live webcam
faces against the known embeddings using cosine distance.
"""

import os
import pickle

import cv2
import numpy as np
from deepface import DeepFace

from utils import DATASET_DIR, ENCODINGS_PATH, logger, ensure_directories

# ---------------------------------------------------------------------------
# Model configuration - shared with attendance.py. Keep these two files in
# sync: if you change EMBEDDING_MODEL here, retrain before running
# attendance, since old encodings from a different model are not comparable.
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "Facenet"       # 128-d embeddings; good speed/accuracy trade-off
DETECTOR_BACKEND = "skip"         # faces are already cropped by face_detector.py


class FaceTrainerError(Exception):
    """Raised when training fails, e.g. no dataset images are found."""
    pass


class FaceTrainer:
    """Builds face embeddings from the images stored in dataset/ using DeepFace."""

    def __init__(self):
        ensure_directories()

    def train(self, progress_callback=None):
        """
        Iterate over every sub-folder of dataset/, compute a DeepFace
        embedding for each image, and save the (encodings, roll_numbers)
        data to disk.

        Folder naming convention: <roll_number>_<name>

        progress_callback(done_folders, total_folders) is optionally invoked
        after each student folder is processed so the GUI can show progress.
        """
        if not os.path.isdir(DATASET_DIR) or not os.listdir(DATASET_DIR):
            raise FaceTrainerError("No images found in dataset/. Register a student first.")

        known_encodings = []
        known_roll_numbers = []

        student_folders = [
            f for f in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, f))
        ]
        if not student_folders:
            raise FaceTrainerError("No student folders found inside dataset/.")

        total_folders = len(student_folders)

        for idx, folder in enumerate(student_folders, start=1):
            roll_number = folder.split("_")[0]
            folder_path = os.path.join(DATASET_DIR, folder)
            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                try:
                    image_bgr = cv2.imread(image_path)
                    if image_bgr is None:
                        continue

                    embedding = self._get_embedding(image_bgr)
                    if embedding is None:
                        continue

                    known_encodings.append(embedding)
                    known_roll_numbers.append(roll_number)
                except Exception as exc:  # noqa: BLE001 - log and continue
                    logger.warning(f"Skipping {image_path}: {exc}")
                    continue

            if progress_callback:
                progress_callback(idx, total_folders)

        if not known_encodings:
            raise FaceTrainerError(
                "Could not generate any face encodings. Try recapturing images with better lighting."
            )

        data = {
            "encodings": known_encodings,
            "roll_numbers": known_roll_numbers,
            "model_name": EMBEDDING_MODEL,
        }
        with open(ENCODINGS_PATH, "wb") as f:
            pickle.dump(data, f)

        logger.info(
            f"Training complete: {len(known_encodings)} encodings from "
            f"{len(student_folders)} students saved to {ENCODINGS_PATH}"
        )
        return len(known_encodings), len(student_folders)

    @staticmethod
    def _get_embedding(image_bgr):
        """Return a single L2-normalized embedding (np.ndarray) for a face crop, or None."""
        try:
            reps = DeepFace.represent(
                img_path=image_bgr,
                model_name=EMBEDDING_MODEL,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"DeepFace could not embed image: {exc}")
            return None

        if not reps:
            return None

        embedding = np.array(reps[0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None
        return embedding / norm
