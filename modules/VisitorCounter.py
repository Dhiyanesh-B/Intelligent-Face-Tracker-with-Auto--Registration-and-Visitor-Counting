"""
Unique Visitor Counter Module — Stable Version
Guards against false double-entry caused by missed face frames.
"""

import logging
from typing import Dict, Set, Optional
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VisitorCounter:
    """
    Maintains accurate count of unique visitors.

    Key guarantees:
      - A face_id can only have ONE active entry open at a time.
      - register_exit() is only effective if register_entry() was called first.
      - Re-identification of the same face_id does NOT create a new entry record
        while a prior entry is still open.
    """

    def __init__(self, db_manager):
        self.db_manager = db_manager

        # track_id → face_id  (active tracks currently inside the frame)
        self.active_visitors: Dict[int, str] = {}

        # face_ids that have ever been seen (for unique count)
        self.unique_visitors: Set[str] = set()

        # face_id → entry timestamp  (open entry events)
        self.entry_times: Dict[str, datetime] = {}

        # face_id → latest exit timestamp
        self.exit_times: Dict[str, datetime] = {}

        # face_ids currently "inside" (have open entry, no matching exit yet)
        self._inside: Set[str] = set()

        self._load_existing()
        logger.info("VisitorCounter initialized.")

    def _load_existing(self):
        """Restore unique visitor set from database on startup."""
        try:
            self.db_manager._execute("SELECT face_id FROM unique_visitors")
            for (fid,) in self.db_manager.cursor.fetchall():
                self.unique_visitors.add(fid)
            logger.info(f"Restored {len(self.unique_visitors)} unique visitors from DB.")
        except Exception as e:
            logger.warning(f"Could not load visitor history: {e}")

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def register_entry(
        self,
        face_id: str,
        track_id: int,
        image: Optional[np.ndarray] = None,
        body_image: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Register an entry event for a confirmed face_id.

        Returns True if this is a brand-new unique visitor.
        Entry is silently ignored if this face_id already has an open entry.
        """
        # Avoid duplicate open entries for the same face
        if face_id in self._inside:
            self.active_visitors[track_id] = face_id
            return False

        timestamp = datetime.now()
        is_new = face_id not in self.unique_visitors

        # Record in memory
        self.unique_visitors.add(face_id)
        self.active_visitors[track_id] = face_id
        self.entry_times[face_id] = timestamp
        self._inside.add(face_id)

        # Save face snapshot
        image_path = None
        if image is not None and image.size > 0:
            image_path = self._save_image(face_id, 'entry', image, timestamp, suffix='face')

        body_path = None
        if body_image is not None and body_image.size > 0:
            body_path = self._save_image(face_id, 'entry', body_image, timestamp, suffix='body')

        # Persist to database
        self.db_manager.log_event(
            face_id, 'entry', image_path,
            confidence=None,
            metadata={
                'track_id': track_id,
                'is_new': is_new,
                'body_image_path': body_path,
            }
        )

        if is_new:
            logger.info(f"NEW visitor: {face_id}  (total unique={len(self.unique_visitors)})")
        else:
            logger.info(f"RETURNING visitor: {face_id}")

        return is_new

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def register_exit(self, track_id: int, exit_reason: str = 'edge') -> Optional[str]:
        """
        Register an exit event for the given track_id.
        Only fires if there is a matching open entry.
        """
        face_id = self.active_visitors.get(track_id)
        if face_id is None:
            return None

        if face_id not in self._inside:
            # Entry was already closed
            self.active_visitors.pop(track_id, None)
            return None

        timestamp = datetime.now()

        # Dwell time
        dwell = None
        if face_id in self.entry_times:
            dwell = (timestamp - self.entry_times[face_id]).total_seconds()

        # Persist to database
        self.db_manager.log_event(
            face_id, 'exit', None,
            metadata={
                'track_id': track_id,
                'dwell_time_seconds': round(dwell, 2) if dwell else None,
                'exit_reason': exit_reason,
            }
        )

        # Close the entry
        self.exit_times[face_id] = timestamp
        self._inside.discard(face_id)
        self.active_visitors.pop(track_id, None)

        dwell_str = f"{dwell:.1f}s" if dwell else "unknown"
        logger.info(f"EXIT: {face_id} | dwell={dwell_str} | reason={exit_reason}")
        return face_id

    # ------------------------------------------------------------------
    # Flush active visitors on shutdown
    # ------------------------------------------------------------------

    def flush_all_exits(self):
        """Call on shutdown to close any still-open entries."""
        for track_id in list(self.active_visitors.keys()):
            self.register_exit(track_id, exit_reason='shutdown')

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_unique_count(self) -> int:
        return len(self.unique_visitors)

    def get_active_count(self) -> int:
        return len(self._inside)

    def get_statistics(self) -> Dict:
        dwell_times = []
        for fid, exit_t in self.exit_times.items():
            if fid in self.entry_times:
                dwell_times.append((exit_t - self.entry_times[fid]).total_seconds())
        avg_dwell = sum(dwell_times) / len(dwell_times) if dwell_times else 0.0

        return {
            'total_unique_visitors': self.get_unique_count(),
            'active_visitors': self.get_active_count(),
            'average_dwell_time_seconds': avg_dwell,
            'total_entries_logged': len(self.entry_times),
            'total_exits_logged': len(self.exit_times),
        }

    # ------------------------------------------------------------------
    # Internal image saving
    # ------------------------------------------------------------------

    def _save_image(
        self,
        face_id: str,
        event_type: str,
        image: np.ndarray,
        timestamp: datetime,
        suffix: str = 'face',
    ) -> Optional[str]:
        try:
            log_cfg = self.db_manager.full_config.get('logging', {})
            base_dir = log_cfg.get('image_storage', 'logs/entries')
            date_dir = Path(base_dir) / timestamp.strftime('%Y-%m-%d')
            date_dir.mkdir(parents=True, exist_ok=True)

            ts_str = timestamp.strftime('%H%M%S_%f')[:10]
            filename = f"{face_id}_{event_type}_{suffix}_{ts_str}.jpg"
            filepath = date_dir / filename

            cv2.imwrite(str(filepath), image)
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save {suffix} image: {e}")
            return None