#!/usr/bin/env python3
"""
AI-Driven Unique Visitor Counter
Architecture:
  1. PersonDetector (YOLO11n)   → stable full-body bounding boxes for tracking
  2. FaceTracker (IoU)          → track persons across frames, survive occlusions
  3. FaceRecognizer (InsightFace)→ RetinaFace detect + ArcFace embed in one call
  4. VisitorCounter             → deduplicate, log entry/exit with guard
  5. DatabaseManager            → persist to PostgreSQL
  6. LoggerSystem               → events.log + face/body snapshots
"""

import cv2
import time
import signal
import sys
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Set
import logging

from modules import (
    FaceRecognizer, FaceTracker,
    PersonDetector, DatabaseManager, LoggerSystem, VisitorCounter
)
from utils.config_loader import ConfigLoader
from utils.image_utils import ImageUtils


def setup_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


logger = logging.getLogger(__name__)


class FaceTrackerApp:
    """
    Main application orchestrator.

    Pipeline per frame:
      ┌─ PersonDetector.detect()  ──► FaceTracker.update()   (every N frames)
      │                                        │
      │  For each confirmed track:             │
      │    crop body from frame                │
      │    if face_id unknown:                 │
      │      FaceRecognizer.detect_and_embed() │  ← InsightFace does face
      │        → RetinaFace detection            │    detection + ArcFace
      │        → ArcFace embedding               │    embedding in one call
      │      FaceRecognizer.recognize_face()   │
      │      if new → register + entry event   │
      │      if known → link ID                │
      │    draw bbox + label                   │
      │                                        │
      └─ For each just_died track:             │
           VisitorCounter.register_exit()      │
    """

    def __init__(self, config_path: str = 'config.json'):
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config

        log_level = self.config.get('logging', {}).get('log_level', 'INFO')
        setup_logging(log_level)

        logger.info("Initializing Face Tracker Application …")
        self.person_detector = PersonDetector(self.config)
        self.face_recognizer = FaceRecognizer(self.config)
        self.db_manager      = DatabaseManager(self.config)
        self.logger_system   = LoggerSystem(self.config)
        self.visitor_counter = VisitorCounter(self.db_manager)
        self.face_tracker    = FaceTracker(self.config)

        self.cap = None
        self.frame_count         = 0
        self.last_detection_frame = -999  # force detection on frame 0
        self.frames_to_skip      = self.config.get('person_detection', {}).get('frames_to_skip',
                                        self.config.get('detection', {}).get('frames_to_skip', 5))

        # Performance: downscale for processing
        self.process_max_dim = self.config['system'].get('process_max_dim', 960)
        self._scale = 1.0

        # Stats
        self.total_frames = 0
        self.fps = 0
        self._fps_cnt  = 0
        self._fps_time = time.time()

        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)

        self._load_face_registry()
        logger.info("Application ready.")

    # ------------------------------------------------------------------
    # Start-up helpers
    # ------------------------------------------------------------------

    def _signal_handler(self, sig, frame):
        logger.info("Shutdown requested …")
        self.running = False

    def _load_face_registry(self):
        try:
            self.db_manager._execute("SELECT face_id FROM unique_visitors")
            count = 0
            for (fid,) in self.db_manager.cursor.fetchall():
                emb = self.db_manager.load_embedding(fid)
                if emb is not None:
                    self.face_recognizer.register_face(fid, emb, {})
                    count += 1
            logger.info(f"Loaded {count} face embeddings from database.")
        except Exception as e:
            logger.warning(f"Could not load face registry: {e}")

    def initialize_video(self, source_type='file', source_path=None) -> bool:
        if source_path is None:
            source_path = (
                self.config['video_source'].get('rtsp_url')
                if source_type == 'rtsp'
                else self.config['video_source'].get('path')
            )
        
        cap = cv2.VideoCapture(source_path)
        if cap is None or not cap.isOpened():
            logger.error(f"Cannot open video: {source_path}")
            return False
            
        self.cap = cap
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._scale = min(1.0, self.process_max_dim / max(w, h))
        logger.info(f"Video: {source_path}  {w}×{h} @ {fps:.1f} FPS  process_scale={self._scale:.2f}")
        return True

    # ------------------------------------------------------------------
    # Frame processing helpers
    # ------------------------------------------------------------------

    def _small(self, frame):
        """Downscale frame for person detection inference."""
        if self._scale >= 1.0:
            return frame
        return cv2.resize(frame, (int(frame.shape[1]*self._scale), int(frame.shape[0]*self._scale)))

    def _up_bbox(self, bbox):
        """Map small-frame bbox back to original resolution."""
        if self._scale >= 1.0:
            return bbox
        inv = 1.0 / self._scale
        return [int(c * inv) for c in bbox]

    def _crop(self, frame, bbox, pad=0):
        """Safely crop bbox from frame with optional padding."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        return frame[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    # Main per-frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return None

        self.total_frames += 1
        self._fps_cnt += 1
        now = time.time()
        if now - self._fps_time >= 1.0:
            self.fps = self._fps_cnt
            self._fps_cnt = 0
            self._fps_time = now

        small = self._small(frame)

        # ── Person Detection (every N frames) ──────────────────────────
        detections = []
        if self.frame_count - self.last_detection_frame >= self.frames_to_skip:
            detections = self.person_detector.detect(small)
            self.last_detection_frame = self.frame_count

        # ── Tracking ───────────────────────────────────────────────────
        tracks = self.face_tracker.update(detections, small.shape[:2])

        # ── Process exits for tracks that just died ────────────────────
        for dead in self.face_tracker.just_died:
            face_id = self.visitor_counter.active_visitors.get(dead['id'])
            if face_id:
                exit_reason = dead.get('exit_reason', 'unknown')
                self.visitor_counter.register_exit(dead['id'], exit_reason)

        # ── Process each active track ──────────────────────────────────
        for track_id, track in tracks.items():
            orig_bbox = self._up_bbox(track['bbox'])
            body_crop = self._crop(frame, orig_bbox, pad=10)

            if body_crop.size == 0:
                continue

            # Assign face ID once per track
            if track['face_id'] is None:
                face_results = []
                
                # Adaptive Cooldown logic:
                # Calculate height relative to frame
                box_h = orig_bbox[3] - orig_bbox[1]
                frame_h = frame.shape[0]
                h_ratio = box_h / frame_h
                
                # Close persons (ratio > 0.35) -> Try often (5 frames)
                # Far persons (ratio < 0.15) -> Try rarely (40 frames)
                # Medium -> 15 frames
                if h_ratio > 0.35:
                    cooldown = 5
                elif h_ratio < 0.15:
                    cooldown = 40
                else:
                    cooldown = 15
                
                last_try = track.get('last_rec_frame', -999)
                if self.frame_count - last_try >= cooldown:
                    track['last_rec_frame'] = self.frame_count
                    # Use InsightFace to detect face + generate embedding in one call
                    face_results = self.face_recognizer.detect_and_embed(body_crop)
                
                if face_results:
                    best = face_results[0]  # highest-confidence face
                    embedding = best['embedding']
                    face_crop = best.get('face_crop')

                    face_id, similarity = self.face_recognizer.recognize_face(embedding)
                    if face_id is None:
                        # New visitor
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]
                        face_id = f"visitor_{ts}_{track_id}"
                        self.face_recognizer.register_face(face_id, embedding, {
                            'first_seen': datetime.now().isoformat(),
                        })
                        self.db_manager.save_embedding(face_id, embedding)
                        self.logger_system.log_recognition(face_id, 0.0, is_new=True)
                    else:
                        self.logger_system.log_recognition(face_id, similarity, is_new=False)

                    track['face_id'] = face_id
                    self.face_tracker.assign_face_id(track_id, face_id)

                    # Register entry (guard inside VisitorCounter prevents duplicates)
                    self.visitor_counter.register_entry(
                        face_id, track_id,
                        image=face_crop,
                        body_image=body_crop,
                    )
            else:
                # Track already identified — ensure entry stays registered
                face_id = track['face_id']
                if face_id not in self.visitor_counter._inside:
                    self.visitor_counter.register_entry(
                        face_id, track_id,
                        body_image=body_crop,
                    )

            # ── Draw ───────────────────────────────────────────────────
            face_id = track.get('face_id')
            x1, y1, x2, y2 = orig_bbox
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)

            if face_id:
                color = (0, 255, 0) if face_id in self.visitor_counter.unique_visitors else (0, 180, 255)
                label = f"{face_id[-8:]}"
            else:
                color = (200, 200, 200)
                label = f"track:{track_id}"

            # Draw body box + confidence
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            conf_txt = f"{track['confidence']:.2f}"
            cv2.putText(frame, label,    (x1, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
            cv2.putText(frame, label,    (x1, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,   1)
            cv2.putText(frame, conf_txt, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            cv2.putText(frame, conf_txt, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,   1)

        # ── Stats overlay ──────────────────────────────────────────────
        self._draw_stats(frame)
        self.frame_count += 1
        return frame

    def _draw_stats(self, frame: np.ndarray):
        lines = [
            f"Unique Visitors: {self.visitor_counter.get_unique_count()}",
            f"Active Now     : {self.visitor_counter.get_active_count()}",
            f"FPS            : {self.fps}",
            f"Frames         : {self.total_frames}",
        ]
        for i, txt in enumerate(lines):
            y = 32 + i * 30
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0),     3)
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 1)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, source_type='file', source_path=None):
        if not self.initialize_video(source_type, source_path):
            return

        display = self.config['system'].get('display_output', True)
        max_disp = 1280

        output_writer = None
        if self.config['system'].get('save_video_output', False):
            out_path = self.config['system']['output_video_path']
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fps_src = self.cap.get(cv2.CAP_PROP_FPS) or 25
            ow = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            oh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_src, (ow, oh))
            logger.info(f"Saving output to: {out_path}")

        logger.info("Processing started — press 'q' to quit.")
        
        target_fps = 30.0
        frame_delay = 1.0 / target_fps
        
        try:
            while self.running:
                loop_start = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("End of stream.")
                    break

                processed = self.process_frame(frame)
                if processed is None:
                    continue

                if display:
                    h, w = processed.shape[:2]
                    if max(h, w) > max_disp:
                        s = max_disp / max(h, w)
                        disp = cv2.resize(processed, (int(w*s), int(h*s)))
                    else:
                        disp = processed
                    cv2.imshow('Visitor Counter', disp)
                    
                    elapsed = time.time() - loop_start
                    delay_ms = max(1, int((frame_delay - elapsed) * 1000))
                    
                    if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                        logger.info("User quit.")
                        break
                else:
                    elapsed = time.time() - loop_start
                    delay_s = frame_delay - elapsed
                    if delay_s > 0:
                        time.sleep(delay_s)

                if output_writer:
                    output_writer.write(processed)

        except KeyboardInterrupt:
            logger.info("Interrupted.")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
        finally:
            # Flush all remaining open entries as exits
            self.visitor_counter.flush_all_exits()
            if output_writer:
                output_writer.release()
            self.cleanup()

    def cleanup(self):
        logger.info("Cleaning up …")
        s = self.visitor_counter.get_statistics()
        logger.info("=== Final Statistics ===")
        logger.info(f"  Unique visitors  : {s['total_unique_visitors']}")
        logger.info(f"  Avg dwell time   : {s['average_dwell_time_seconds']:.1f}s")
        logger.info(f"  Entries logged   : {s['total_entries_logged']}")
        logger.info(f"  Exits logged     : {s['total_exits_logged']}")
        self.db_manager.close()
        self.logger_system.close()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Done.")


# ------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description='AI Visitor Counter')
    p.add_argument('--config', default='config.json')
    p.add_argument('--source', choices=['file', 'rtsp'], default='file')
    p.add_argument('--path', help='Video file or RTSP URL')
    p.add_argument('--no-display', action='store_true')
    args = p.parse_args()

    app = FaceTrackerApp(args.config)
    if args.no_display:
        app.config['system']['display_output'] = False
    app.run(source_type=args.source, source_path=args.path)


if __name__ == '__main__':
    main()