"""
Image Processing Utilities
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class ImageUtils:
    """
    Utility functions for image processing
    """
    
    @staticmethod
    def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: Input image
            max_size: Maximum dimension size
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        
        if max(h, w) <= max_size:
            return image
        
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(image, (new_w, new_h))
    
    @staticmethod
    def draw_bbox(image: np.ndarray, bbox: List[int], color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2, label: str = None) -> np.ndarray:
        """
        Draw bounding box on image
        
        Args:
            image: Input image
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            color: RGB color tuple
            thickness: Line thickness
            label: Optional label text
            
        Returns:
            Image with bounding box drawn
        """
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label if provided
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(image, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        
        return image
    
    @staticmethod
    def enhance_face(face_crop: np.ndarray) -> np.ndarray:
        """
        Enhance face image for better recognition
        
        Args:
            face_crop: Face crop image
            
        Returns:
            Enhanced face image
        """
        # Apply histogram equalization
        if len(face_crop.shape) == 3:
            # Convert to YUV for better enhancement
            yuv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            enhanced = cv2.equalizeHist(face_crop)
        
        return enhanced
    
    @staticmethod
    def extract_face_landmarks(face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract facial landmarks using dlib (if available)
        
        Args:
            face_image: Face image
            
        Returns:
            Array of landmark points or None
        """
        try:
            import dlib
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            if len(faces) > 0:
                landmarks = predictor(gray, faces[0])
                points = np.array([[p.x, p.y] for p in landmarks.parts()])
                return points
            
        except ImportError:
            logger.debug("dlib not available for landmark detection")
        except Exception as e:
            logger.error(f"Landmark extraction failed: {e}")
        
        return None
    
    @staticmethod
    def align_face(face_image: np.ndarray, landmarks: np.ndarray = None) -> np.ndarray:
        """
        Align face based on eye positions
        
        Args:
            face_image: Face image
            landmarks: Facial landmarks (optional)
            
        Returns:
            Aligned face image
        """
        if landmarks is None:
            landmarks = ImageUtils.extract_face_landmarks(face_image)
        
        if landmarks is None or len(landmarks) < 36:
            return face_image
        
        # Get eye coordinates (assuming 68 point landmarks)
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        
        # Calculate angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Rotate image
        (h, w) = face_image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(face_image, rotation_matrix, (w, h))
        
        return aligned