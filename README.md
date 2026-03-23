# Intelligent Face Tracker & Visitor Counter

A production-ready edge AI application for robust, real-time person tracking and face recognition. It maintains accurate unique visitor counts, calculates dwell times, and logs all entry and exit events with body and face snapshots.

## 🚀 Key Features and Architecture

The system has been heavily optimized for real-world scenarios where faces turn away, become occluded, or disappear briefly.
To achieve this, it relies on a **body-first tracking** approach coupled with an **Adaptive Cooldown Mechanism** for face scanning.

- **Person Detection (YOLOv26)**: A specialized YOLOv26 detection layer targeting the *person* class to establish stable full-body tracking boxes, surviving head turns and partial occlusions.
- **Face Recognition (InsightFace)**: Employs `RetinaFace` for face detection within body crops and `ArcFace` for extracting high-dimensional biometrics. Runs dynamically via Adaptive Cooldowns.
- **Tracking Layer (Kalman + IoU)**: Maintains object permanence (ghost tracks up to 60 frames) making it highly resilient.
- **Auto-Registration**: Unrecognized faces are assigned unique IDs on-the-fly without manual enrollment.
- **Dwell Time Calculation**: Intelligently handles entry/exit timing based on edge zones and track death reasons.
- **Database Scalability**: Exclusive, native `PostgreSQL` support for concurrent, persistent storage.
- **Turbo Mode Optimization**: Processes on reduced frame dimensions (`process_max_dim`) and configurable skip intervals to maintain high FPS even on CPU.

## 📚 Additional Documentation
- [AI Planning & Workloads](ai_planning.md)
- [Architecture Diagram](architecture.md)

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- PostgreSQL Server 
- CUDA GPU (Optional, but highly recommended for `InsightFace`)

### Installation Steps

1. **Clone & Virtual Environment**
```bash
git clone https://github.com/yourusername/face-tracker-system.git
cd face-tracker-system
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Database & Tracking**
Update `config.json` with your PostgreSQL credentials. It must include `type: "postgresql"`. `config.json.example` (or the existing `config.json`) has the structure.

4. **Run the App**
```bash
# File source
python main.py --source file --path sample_video2.mp4

# RTSP Stream
python main.py --source rtsp --path rtsp://camera_ip
```

## 🛠️ Configuration Guide (`config.json`)

### Sample `config.json` Structure
```json
{
  "video_source": { "type": "file", "path": "sample_video2.mp4", "rtsp_url": "rtsp://..." },
  "person_detection": { "model": "yolo26n.pt", "confidence_threshold": 0.4, "max_detections_per_frame": 20, "frames_to_skip": 10 },
  "recognition": { "model": "buffalo_l", "similarity_threshold": 0.38, "use_gpu": true },
  "tracking": { "max_age": 60, "min_hits": 2, "iou_threshold": 0.3, "min_active_frames": 8, "edge_zone_ratio": 0.15 },
  "database": { "type": "postgresql", "path": "database/face_tracker.db", "postgresql": { "host": "localhost", "port": 5432, "database": "face_tracker", "user": "postgres", "password": "admin" } },
  "system": { "process_every_n_frames": 2, "process_max_dim": 640, "display_output": true }
}
```

The `config.json` encapsulates all operational thresholds.

### Important Tunings:
- **`system.process_every_n_frames`**: Throttles the absolute processing rate (e.g., 2 = processes 15 FPS of a 30 FPS video).
- **`system.process_max_dim`**: Global downscaling dimension (e.g. 640px) to boost performance. All detection boxes map perfectly back to the HD frame.
- **`person_detection.frames_to_skip`**: How often the heavy YOLO model runs (e.g., 10 means detectors run every 10th processed frame, tracking handles the rest).
- **`recognition.similarity_threshold`**: Cosine similarity cutoff (e.g., `0.38`). Lower is more lenient.
- **`tracking.max_age`**: Frames before a lost track is considered dead and triggers an "exit" event (e.g., `60`).
- **`tracking.edge_zone_ratio`**: Defines the border margin (e.g., `0.15`). If a person dies in this zone, they are marked as `exit_reason=edge`.
- **`logging.max_log_size_mb`**: Rotational threshold for `events.log`.

## 🗄️ Database & Storage

*SQLite has been formally deprecated for this architecture. Use PostgreSQL exclusively.*

### Postgres Tables:
- **`face_entries`**: All discrete events (`id`, `face_id`, `event_type`, `timestamp`, `metadata JSONB`).
- **`unique_visitors`**: Aggregated state (`face_id`, `first_seen`, `total_entries`, `total_exits`, `metadata JSONB`).
- **`face_embeddings`**: Serialized embeddings (`BYTEA`).

### Snapshot Storage:
Logged automatically to `/logs/entries/YYYY-MM-DD/`. Both `_body_` and `_face_` crops are saved upon new visitor entry.

## 🤔 Assumptions Made
1. **Camera Input**: Subjects are reasonably sized (>50x50 pixels) for accurate detection.
2. **Lighting Conditions**: Adequate lighting for face recognition (works best in well-lit environments).
3. **Face Orientation**: Frontal or near-frontal faces for optimal recognition biometrics.
4. **Database**: PostgreSQL is strictly required for this production iteration; SQLite is deprecated.
5. **Video Source**: Stable frame rate (assumes consistent FPS from source).
6. **Tracking**: Subjects move continuously between frames (smooth motion).

## ⚡ Performance Breakdown
Through the **Adaptive Cooldown Mechanism**, the recognition engine tests close targets (occupying >35% vertical height) every 5 frames, but distant targets (<15%) every 40 frames. The pipeline scales efficiently on multi-core CPUs (~30FPS) and screams on modern GPUs (>60FPS).

## 🎥 Explanation Video

👉 Click to watch full video 
[Watch explanation](./explanation.mp4)

## License
MIT License.

---
*This project is a part of a hackathon run by https://katomaran.com*
