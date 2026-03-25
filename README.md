# 🔐 Face Recognition V2 — Real-Time Face Verification System

> A production-oriented face verification pipeline built with deep learning, designed for real-world security applications.

---

## 📌 Overview

**Face Recognition V2** is a real-time face verification system that implements a complete machine learning pipeline — from data acquisition to live identity verification.

Unlike basic demos, this project focuses on **system design, automation, and real-world usability**, making it a strong foundation for scalable security solutions.

It serves as the **core engine for the upcoming USB Locker V3**, where facial authentication will control system-level access.

---

## 🚀 Highlights

* ⚡ Real-time face detection & verification
* 🧠 Deep learning pipeline (MTCNN + FaceNet)
* 🔄 Fully automated workflow (no manual setup)
* 📸 Live dataset generation from camera
* ⚙️ Precomputed embeddings for fast inference
* 📱 Mobile camera support via DroidCam
* 🧩 Modular, production-style architecture

---

## 🧠 System Pipeline

```text
Camera → Face Detection → Face Cropping → Embedding → Similarity Check → Decision
```

---

## ⚙️ Smart Self-Initializing System

This system requires **zero manual setup**:

* If no dataset → automatically captures images
* If no embeddings → builds them automatically
* If everything exists → runs instantly

---

## 📂 Project Structure

```
Face_Recognition_V2/
│
├── data/
│   ├── raw/
│   └── embeddings/
│
├── src/
│   ├── pipeline/
│   │   ├── detector.py
│   │   ├── embedder.py
│   │   ├── recognizer.py
│   │   ├── capture_dataset.py
│   │   └── droidCam.py
│   │
│   └── main.py
```

---

## ▶️ Run the Project

```bash
pip install -r requirements.txt
python src/main.py
```

---

## 🔬 Technical Details

* **Detection:** MTCNN
* **Embedding:** FaceNet (keras-facenet)
* **Verification:** Euclidean distance + thresholding
* **Pipeline:** Fully automated (capture → build → run)

---

## ⚡ Performance

* Real-time execution on CPU
* No GPU required
* Efficient and lightweight pipeline
* Stable under different lighting conditions

---

## 🧪 Real-World Constraints

* Built and tested on a **low-end personal machine**
* Designed to work under limited resources
* Focused on practical deployment, not just theory

---

## 📸 Screenshots / Demo

* Face detection in real-time
* Authorized vs Unauthorized result
* Dataset capture phase
* System running live



---

## 🎯 Use Cases

* Secure workstation access
* Face-based authentication systems
* Security prototypes
* Foundation for hardware-integrated solutions

---

## 🔮 Next Step

This project is the **foundation for USB Locker V3**, where face recognition will be integrated with system-level security controls.

---

## 👤 Author

**ATEF Aliat**
Machine Learning & Computer Vision Enthusiast

* Focus: Real-time systems, AI pipelines, and security applications
* Built projects under hardware constraints to simulate real-world conditions
* Strong interest in applied AI and system design

---

## ⭐ Final Statement

This project demonstrates the ability to:

* Design and implement end-to-end ML systems
* Work with real-time data pipelines
* Build practical solutions under constraints

> Not just a model — but a complete, working system.
