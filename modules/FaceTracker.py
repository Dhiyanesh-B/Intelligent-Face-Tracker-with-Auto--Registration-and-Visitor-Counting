"""
Stable Face/Person Tracker using IoU matching.
Key design principles:
  - Large max_age so temporary occlusions or face turns never kill a track.
  - min_active_frames guard: exit only logged when a person was truly present
    for a meaningful number of frames (not just a 1-frame blip).
  - Edge-zone detection: a track is only considered an exit when its last
    known centroid is near the frame edge, preventing interior occlusion
    from being mistaken for an exit.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import logging

logger = logging.getLogger(__name__)


class FaceTracker:
    """
    Robust IoU-based person tracker.
    Separates "entry confirmation" (min_hits) from "exit confirmation"
    (max_age exceeded AND person was genuinely active).
    """

    def __init__(self, config: Dict):
        cfg = config['tracking']
        self.max_age   = cfg.get('max_age', 60)          # frames before track dies
        self.min_hits  = cfg.get('min_hits', 2)           # hits before track is "confirmed"
        self.iou_threshold = cfg.get('iou_threshold', 0.3)
        # Minimum frames a track must have been active to count as a real exit
        self.min_active_frames = cfg.get('min_active_frames', 8)
        # Fraction of frame width/height that counts as "edge zone"
        self.edge_zone = cfg.get('edge_zone_ratio', 0.15)

        self.tracks: Dict[int, Dict] = {}
        self.next_id = 0
        # Tracks that just died this frame (for exit processing in main.py)
        self._just_died: List[Dict] = []

        logger.info(
            f"FaceTracker: max_age={self.max_age}, min_hits={self.min_hits}, "
            f"min_active_frames={self.min_active_frames}, edge_zone={self.edge_zone}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, detections: List[Dict], frame_shape: Tuple[int, int]
    ) -> Dict[int, Dict]:
        """
        Update tracker.
        Returns only confirmed, living tracks (hits >= min_hits, age <= max_age).
        Populates self._just_died with tracks that died this frame.
        """
        fh, fw = frame_shape
        self._just_died = []

        # Age all existing tracks
        for t in self.tracks.values():
            t['age'] += 1

        # Match detections → existing tracks
        matched, unmatched_dets = self._match(detections)

        # Update matched
        for track_id, det_idx in matched:
            det = detections[det_idx]
            t = self.tracks[track_id]
            t['bbox'] = det['bbox']
            t['hits'] += 1
            t['active_frames'] += 1
            t['age'] = 0
            t['confidence'] = det.get('confidence', 1.0)

        # Create new tracks for unmatched detections
        for di in unmatched_dets:
            self._create(detections[di])

        # Prune dead tracks; collect real exits
        dead_ids = [tid for tid, t in self.tracks.items() if t['age'] > self.max_age]
        for tid in dead_ids:
            t = self.tracks.pop(tid)
            # Only a real exit if the person was confirmed active long enough
            if t['active_frames'] >= self.min_active_frames and t['hits'] >= self.min_hits:
                t['exit_reason'] = self._exit_reason(t, fw, fh)
                self._just_died.append(t)
                logger.debug(
                    f"Track {tid} died: active={t['active_frames']} frames, "
                    f"exit_reason={t['exit_reason']}"
                )

        # Return confirmed living tracks
        return {
            tid: t for tid, t in self.tracks.items()
            if t['hits'] >= self.min_hits
        }

    @property
    def just_died(self) -> List[Dict]:
        """Tracks that died this update pass (for exit event logging)."""
        return self._just_died

    def assign_face_id(self, track_id: int, face_id: str):
        if track_id in self.tracks:
            self.tracks[track_id]['face_id'] = face_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create(self, detection: Dict):
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = {
            'id': tid,
            'bbox': detection['bbox'],
            'hits': 1,
            'age': 0,
            'active_frames': 1,
            'confidence': detection.get('confidence', 1.0),
            'face_id': None,
            'entry_bbox': detection['bbox'],   # where this person appeared
        }

    def _match(
        self, detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        if not self.tracks or not detections:
            return [], list(range(len(detections)))

        tids = list(self.tracks.keys())
        iou_mat = np.zeros((len(tids), len(detections)), dtype=np.float32)

        for i, tid in enumerate(tids):
            for j, det in enumerate(detections):
                iou_mat[i, j] = self._iou(self.tracks[tid]['bbox'], det['bbox'])

        matched, used_dets, used_tracks = [], set(), set()
        for idx in np.argsort(-iou_mat.flatten()):
            ti, di = divmod(int(idx), len(detections))
            if ti in used_tracks or di in used_dets:
                continue
            if iou_mat[ti, di] >= self.iou_threshold:
                matched.append((tids[ti], di))
                used_tracks.add(ti)
                used_dets.add(di)

        unmatched = [j for j in range(len(detections)) if j not in used_dets]
        return matched, unmatched

    @staticmethod
    def _iou(b1, b2) -> float:
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def _exit_reason(self, track: Dict, fw: int, fh: int) -> str:
        """Determine if person left via an edge or disappeared mid-frame."""
        x1, y1, x2, y2 = track['bbox']
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        ez = self.edge_zone
        if cx < fw * ez:      return 'left_edge'
        if cx > fw * (1-ez):  return 'right_edge'
        if cy < fh * ez:      return 'top_edge'
        if cy > fh * (1-ez):  return 'bottom_edge'
        return 'occlusion'    # disappeared in middle of frame