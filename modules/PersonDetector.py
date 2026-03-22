"""
Person Detector Module using YOLOv8 (COCO person class)
Used for stable full-body tracking even when face turns or is occluded.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0  # COCO class 0 = person


class PersonDetector:
    """
    Detects full-body person bounding boxes using YOLOv8 (general COCO model).
    Body detection is far more robust than face detection for tracking purposes —
    it survives head turns, partial occlusions, and brief look-aways.
    """

    def __init__(self, config: Dict):
        cfg = config.get('person_detection', {})
        model_path = cfg.get('model', 'yolo26n.pt')
        self.conf = cfg.get('confidence_threshold', 0.4)
        self.iou = cfg.get('iou_threshold', 0.45)
        self.max_det = cfg.get('max_detections_per_frame', 20)

        try:
            self.model = YOLO(model_path)
            logger.info(f"PersonDetector: loaded {model_path}")
        except Exception as e:
            logger.error(f"PersonDetector: failed to load {model_path}: {e}")
            self.model = None
            raise RuntimeError(f"Could not load YOLO model: {e}")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"PersonDetector: using {self.device}")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in a frame.

        Returns:
            List of {'bbox': [x1,y1,x2,y2], 'confidence': float}
        """
        if frame is None or frame.size == 0 or self.model is None:
            return []

        results = self.model(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )

        persons = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                persons.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                })

        logger.debug(f"PersonDetector: {len(persons)} persons detected")
        return persons
