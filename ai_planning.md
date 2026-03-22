# General Project Plan: AI-Powered Face Tracker

Hey! For this hackathon, I want you (the AI) to help me build a production-ready, real-time edge processing node that can track and count individuals. Here is my plan and the instructions you must follow to generate the application.

## 1. Planning the app
My core strategy is to avoid the pitfalls of traditional face detection (which drops tracking the second someone turns their head). Instead, I want to use a **body-first** approach.
- You will need to implement an object detector that tracks **full bodies** first. Use YOLOv26 (`yolo26n.pt`) for this.
- Once a body is tracked, you will extract the face from that body crop and pass it to InsightFace (using `buffalo_l` / ArcFace) to generate a facial embedding.
- We need to handle database storage correctly. I don't want to use SQLite; write the integration strictly for `PostgreSQL`. Use the `ON CONFLICT (face_id) DO UPDATE` pattern so we don't throw errors when seeing the same person twice.
- You must abstract all thresholds (confidence, skip frames, edge zones) into a `config.json` file so I can hot-swap settings without touching the Python code.

## 2. Listing and documenting all the features that the app will perform
Here is the strict feature list I expect you to build into the app:
* **YOLOv26 Person Detection**: Detect persons in real-time to generate bounding boxes localized to full human bodies.
* **Vector Embeddings (InsightFace)**: Use RetinaFace to detect the face within the body crop, and ArcFace to convert it into a highly distinguishable 512-dimensional biometric vector.
* **Kalman Filter Biometric Tracking**: Maintain track states. If a tracking box is lost due to occlusion, you must maintain its memory (`max_age` of 60 frames) and attempt to re-associate it via Intersection over Union (IoU) matching.
* **Auto-Registration Protocol**: If a generated embedding doesn't match our Postgres database (using cosine similarity < 0.38), register them as a new user dynamically and store their embedding in a `BYTEA` column.
* **Event Logging & Snapshotting**: Whenever someone enters or leaves the frame, save an annotated crop of their face AND their body to the local filesystem (e.g. `logs/entries/...`).
* **Unique Dwell & Visitor Counting**: Deduplicate the flow. If someone walks in and stands there for 5 minutes, that counts as 1 entry, 1 dwell time log, and 1 exit.
* **Optimization**: Create an 'Adaptive Cooldown' mechanism. I don't want you running InsightFace 30 times a second on the same person.

## 3. Estimate the amount of compute load consumption in both CPU and GPU
Before we write the code, here is my computation estimate that we need to design around:

### CPU-Only Workload Constraint
* **Detection (YOLOv26):** Will likely take ~30-50ms per frame.
* **Recognition (InsightFace):** Very heavy, around 40-70ms per face.
* **Our Optimization Goal:** Because of this load, we process videos at exactly 640px width, skip every 2nd frame (`process_every_n_frames: 2`), and only run the YOLO detector every 10 tracked frames.
If you build this correctly, we can maintain ~30 FPS on a standard multi-core CPU.

### GPU Workload Expectations
* **Detection (CUDA):** Should take < 5ms.
* **Recognition (CUDA):** Should take < 10ms.
* **Our Optimization Goal:** The architecture must natively pass tensors to the `cuda` device if `torch.cuda.is_available()` is True. If run on an RTX card or T4 cloud VM, we should easily exceed 60+ FPS processing speed.

---
Let's build this! Use modern, clean, modular Python architecture (classes for Database, Tracking, UI, etc.).

## 🎥 Demo Video
[Insert Loom or YouTube video link demonstrating the solution here]

*This project is a part of a hackathon run by https://katomaran.com*
