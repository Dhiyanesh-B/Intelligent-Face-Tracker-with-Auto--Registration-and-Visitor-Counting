"""
Logging System for face events and system operations
"""

import logging
import logging.handlers
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import cv2
import uuid
import numpy as np

class LoggerSystem:
    """
    Centralized logging system for events and image storage
    """
    
    def __init__(self, config: Dict):
        """
        Initialize logging system
        
        Args:
            config: Configuration dictionary containing logging settings
        """
        self.config = config['logging']
        self.log_file = self.config['log_file']
        self.image_storage = self.config['image_storage']
        self.log_level = getattr(logging, self.config['log_level'])
        
        # Create directories
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.image_storage).mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
        
        # Event buffer for JSON logging
        self.event_buffer = []
        
        self.logger.info("Logging system initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup main logger with file and console handlers"""
        logger = logging.getLogger('FaceTracker')
        logger.setLevel(self.log_level)
        
        # File handler with rotation
        max_bytes = self.config.get('max_log_size_mb', 100) * 1024 * 1024
        backup_count = self.config.get('backup_count', 5)
        
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(self.log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_event(self, event_type: str, face_id: str, 
                  event_data: Dict = None, save_image: bool = True,
                  image: Optional[np.ndarray] = None):
        """
        Log a system event with optional image storage
        
        Args:
            event_type: Type of event (entry, exit, recognition, registration, etc.)
            face_id: Face identifier
            event_data: Additional event data
            save_image: Whether to save the face image
            image: Face image to save
        """
        timestamp = datetime.now()
        event_info = {
            'timestamp': timestamp.isoformat(),
            'event_type': event_type,
            'face_id': face_id,
            'data': event_data or {}
        }
        
        # Save image if provided
        image_path = None
        if save_image and image is not None and image.size > 0:
            image_path = self.save_face_image(face_id, event_type, image, timestamp)
            event_info['image_path'] = image_path
        
        # Log to file
        self.logger.info(f"EVENT: {json.dumps(event_info)}")
        
        # Store in buffer for batch processing
        self.event_buffer.append(event_info)
        if len(self.event_buffer) >= 100:
            self.flush_events()
    
    def save_face_image(self, face_id: str, event_type: str, 
                       image: np.ndarray, timestamp: datetime = None) -> str:
        """
        Save face image to local storage
        
        Args:
            face_id: Face identifier
            event_type: Event type (entry, exit, etc.)
            image: Face image as numpy array
            timestamp: Event timestamp
            
        Returns:
            Path to saved image
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create directory structure: YYYY-MM-DD/
        date_dir = timestamp.strftime('%Y-%m-%d')
        full_dir = Path(self.image_storage) / date_dir
        full_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename: face_id_eventtype_timestamp_uuid.jpg
        timestamp_str = timestamp.strftime('%H%M%S_%f')[:-3]
        filename = f"{face_id}_{event_type}_{timestamp_str}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = full_dir / filename
        
        # Save image
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                # OpenCV uses BGR, convert to RGB for saving
                save_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                save_image = image
            
            cv2.imwrite(str(filepath), save_image)
            self.logger.debug(f"Saved face image: {filepath}")
            return str(filepath)
        except Exception as e:
            self.logger.error(f"Failed to save face image: {e}")
            return None
    
    def log_recognition(self, face_id: str, confidence: float, 
                        is_new: bool = False, metadata: Dict = None):
        """
        Log face recognition event
        
        Args:
            face_id: Face identifier
            confidence: Recognition confidence score
            is_new: Whether this is a new face registration
            metadata: Additional metadata
        """
        event_data = {
            'confidence': confidence,
            'is_new': is_new,
            'metadata': metadata or {}
        }
        
        event_type = 'registration' if is_new else 'recognition'
        self.log_event(event_type, face_id, event_data, save_image=False)
    
    def log_tracking(self, face_id: str, track_id: int, 
                     action: str, bbox: list = None):
        """
        Log face tracking event
        
        Args:
            face_id: Face identifier
            track_id: Tracking ID
            action: Tracking action (start, update, end)
            bbox: Bounding box coordinates
        """
        event_data = {
            'track_id': track_id,
            'action': action,
            'bbox': bbox
        }
        
        self.log_event('tracking', face_id, event_data, save_image=False)
    
    def flush_events(self):
        """Flush event buffer to file"""
        if self.event_buffer:
            # Could implement batch writing to database here
            self.event_buffer.clear()
            self.logger.debug(f"Flushed {len(self.event_buffer)} events")
    
    def get_event_summary(self, face_id: str = None) -> Dict:
        """
        Get event summary for a face or all faces
        
        Args:
            face_id: Optional face identifier to filter by
            
        Returns:
            Dictionary with event statistics
        """
        # This would typically query the database
        # For now, return basic structure
        return {
            'total_events': len(self.event_buffer),
            'events_by_type': {},
            'faces': []
        }
    
    def close(self):
        """Close logger and flush remaining events"""
        self.flush_events()
        
        # Close all handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        
        self.logger.info("Logging system closed")