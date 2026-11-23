# 🎯 Face Recognition Attendance System

A desktop attendance management system that uses **real-time face recognition**
to automatically mark student attendance, built with Python, OpenCV,
[DeepFace](https://github.com/serengil/deepface), SQLite, and Tkinter.

> This version replaces `dlib` / `face_recognition` entirely with
> **DeepFace + OpenCV**, so there is no CMake / Visual Studio Build Tools
> step required to install it on Windows.

This project was designed to be **resume-ready** and demonstrates practical,
interview-friendly skills — computer vision, GUI development, database design,
and clean, modular OOP code — the kind of applied project commonly showcased
in placements for **Cognizant GenC, Infosys, TCS, Accenture, and Capgemini**.

---

## 📌 Project Overview

Traditional roll-call or biometric-card attendance is slow and easy to
manipulate ("proxy attendance"). This system solves that by:

1. Registering each student's face through the webcam.
2. Training a face-recognition model on the captured images.
3. Running real-time recognition during a class/session.
4. Automatically marking attendance in a database the moment a registered
   face is detected — while preventing duplicate entries for the same day.

Everything is wrapped in a clean Tkinter desktop GUI with a dashboard,
registration form, live attendance window, and reporting tools.

---

## ✨ Features

- 🔐 **Login screen** to gate access to the application
- 🧑‍🎓 **Register new student** with name, roll number, department, and email
- 📸 **Capture face images** via webcam (auto-cropped using Haar Cascade)
- 🧠 **Train face recognition model** (embeddings via DeepFace's `Facenet` model)
- 👁️ **Real-time face recognition** using a live webcam feed (OpenCV detection + DeepFace embeddings)
- ✅ **Automatic attendance marking** the instant a known face is recognized
- 🚫 **Duplicate-attendance prevention** — one entry per student per day
- 🔍 **Search attendance** by student name or roll number
- 📋 **View attendance records** (all-time or today-only)
- 📤 **Export attendance to CSV** for reporting
- 🗑️ **Delete student** (cascades and removes their attendance history)
- ✏️ **Update student information**
- 📊 **Dashboard** showing total students, present today, and absent today

---

## 🛠 Technologies Used

| Layer               | Technology                          |
|---------------------|--------------------------------------|
| Language            | Python 3.11+                         |
| Computer Vision     | OpenCV (Haar Cascade detection)      |
| Face Recognition    | DeepFace (`Facenet` embeddings, TensorFlow backend) |
| GUI                 | Tkinter (`ttk` widgets)               |
| Database            | SQLite3                              |
| Data handling       | Pandas, NumPy                        |
| Image export/report | CSV via Pandas                       |

---

## 📁 Folder Structure

```
FaceRecognitionAttendance/
│
├── database/
│   └── attendance.db          # Auto-created SQLite database
│
├── dataset/                   # Captured face images, one folder per student
│   └── <roll_number>_<name>/
│
├── trained_model/
│   └── encodings.pickle       # Serialized DeepFace embeddings (after training)
│
├── attendance/                # Exported CSV attendance reports
│
├── screenshots/               # Auto-saved snapshot at each attendance mark
│
├── src/
│   ├── face_detector.py       # Webcam face capture (OpenCV Haar Cascade)
│   ├── face_trainer.py        # Generates DeepFace embeddings from dataset/
│   ├── attendance.py          # Real-time recognition (OpenCV + DeepFace) + attendance marking
│   ├── database.py            # SQLite database access layer (OOP)
│   ├── gui.py                 # All Tkinter windows/screens
│   ├── utils.py               # Validation, paths, logging helpers
│   └── main.py                # Application entry point
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🗄 Database Schema

**students**

| Column       | Type    | Notes                  |
|--------------|---------|------------------------|
| id           | INTEGER | Primary key, autoincrement |
| name         | TEXT    | Required               |
| roll_number  | TEXT    | Unique, required        |
| department   | TEXT    | Required               |
| email        | TEXT    | Required                |

**attendance**

| Column         | Type    | Notes                              |
|----------------|---------|-------------------------------------|
| attendance_id  | INTEGER | Primary key, autoincrement          |
| student_id     | INTEGER | Foreign key → students.id           |
| date           | TEXT    | YYYY-MM-DD                          |
| time           | TEXT    | HH:MM:SS                            |
| status         | TEXT    | 'Present' / 'Absent'                |

---

## ⚙️ Installation Steps

### 1. Clone / Download the project
```bash
git clone <your-repo-url>
cd FaceRecognitionAttendance
```

### 2. Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

No CMake, no Visual Studio Build Tools, no `dlib` compilation — every
dependency here (OpenCV, DeepFace, TensorFlow) ships as a prebuilt wheel on
PyPI for Windows, so a plain `pip install -r requirements.txt` is enough.

> 💡 **First run will download model weights.** The first time you train
> or run attendance, DeepFace automatically downloads the `Facenet` model
> weights (a few MB) to `~/.deepface/weights/` on your machine. This
> requires an internet connection the first time only; after that it's
> cached locally.

---

## ▶️ How to Run

From the project root:

```bash
python src/main.py
```

**Default login credentials:**
- Username: `admin`
- Password: `admin123`

### Typical workflow
1. **Login** with the credentials above.
2. From the **Dashboard**, click **Register Student** → fill the form →
   click **Capture Face & Register** (webcam opens and captures ~40 images).
3. Click **Train Model** to generate face encodings.
4. Go back to the Dashboard → click **Take Attendance** → click
   **Start Recognition**. Recognized students are marked present automatically.
5. Use **View Attendance** to search, filter by today, or export to CSV.
6. Use **Manage Students** to update or delete student records.

---

## 🚀 Future Enhancements

- Email/SMS notifications to students/parents on attendance marked
- Web-based dashboard (Flask/Django + React) alongside the desktop app
- Cloud database (PostgreSQL/MySQL) for multi-device/lab deployments
- Face mask / liveness detection to prevent photo-spoofing
- Role-based login (Admin / Faculty) with proper password hashing
- REST API for integration with existing college ERP systems
- Attendance analytics (monthly trends, department-wise reports, charts)

---

## 👤 Author

**Your Name**
Feel free to connect and share feedback!
- GitHub: [your-github-profile](https://github.com/psesaank)
- LinkedIn: [your-linkedin-profile]([https://linkedin.com/in/your-profile](https://www.linkedin.com/in/sesaank-potharlanka-702306320/))
- Email: sesaankpotharlanka866@gmail.com

---

## 📄 License

This project is open-source and available under the MIT License — free to
use and modify for learning and academic purposes.
