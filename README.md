# Face Recognition Attendance System

A Python-based **Face Recognition Attendance Management System** that automates student identification and attendance tracking using computer vision and deep learning.

The system uses OpenCV for face detection and DeepFace with the FaceNet model for facial recognition, helping simplify the attendance process through a graphical user interface.

## Overview

Traditional attendance systems require manual effort and can be time-consuming. This project provides an automated approach by recognizing registered students through a webcam and recording their attendance.

## Features

- **Face Detection:** Detects faces using OpenCV.
- **Face Recognition:** Identifies registered students using DeepFace and FaceNet.
- **Automated Attendance:** Records attendance after successful face recognition.
- **Student Registration:** Supports collecting student face images for recognition.
- **Model Training:** Generates facial embeddings from registered student images.
- **Graphical User Interface:** Provides an interface for interacting with the system.
- **Attendance Management:** Stores attendance records using a database.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Face detection and camera processing |
| DeepFace | Facial recognition |
| FaceNet | Face embedding generation |
| TensorFlow | Deep learning framework |
| Tkinter | Graphical user interface |
| SQLite | Attendance data storage |

## Project Structure

```text
Face-Recognition-Attendance-System/
│
├── src/
│   ├── attendance.py
│   ├── database.py
│   ├── face_detector.py
│   ├── face_trainer.py
│   ├── gui.py
│   ├── main.py
│   └── utils.py
│
├── attendance/
├── database/
├── dataset/
├── trained_model/
├── screenshots/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/mahaznoor/Face-Recognition-Attendance-System.git
```

### 2. Navigate to the Project Directory

```bash
cd Face-Recognition-Attendance-System
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python src/main.py
```

## How It Works

1. Launch the application.
2. Register students and capture their face images.
3. Train the face recognition model using the registered images.
4. Start the attendance process using the webcam.
5. The system detects and recognizes registered faces.
6. Attendance is recorded for recognized students.

## Requirements

- Python 3.11 recommended
- Webcam
- Windows, Linux, or macOS
- Working internet connection for initial dependency installation

## Privacy & Data Protection

This repository does not include actual student face images, trained facial embeddings, or personal attendance records.

Face images and facial recognition data are biometric information. Keep them secure, obtain appropriate consent before collecting them, and never publish real student data in a public repository.

## Future Improvements

- Improve recognition accuracy and reduce false matches.
- Add confidence and ambiguity checks.
- Introduce attendance reports and export functionality.
- Add student management features.
- Improve the graphical user interface.

## Author

**Mahaz Noor**

Computer Science Graduate | Python | Machine Learning | AI | NLP | React

- GitHub: [@mahaznoor](https://github.com/mahaznoor)
- Project Repository: [Face Recognition Attendance System](https://github.com/mahaznoor/Face-Recognition-Attendance-System)

---
