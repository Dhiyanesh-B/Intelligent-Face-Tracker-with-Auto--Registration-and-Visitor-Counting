"""
Configuration Loader Utility
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Load and manage configuration from JSON file
    """
    
    def __init__(self, config_path: str = 'config.json'):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Validate configuration
        self._validate_config()
        
        logger.info("Configuration loaded successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
    
    def _validate_config(self):
        """Validate required configuration fields"""
        required_sections = ['person_detection', 'recognition', 'tracking', 'database', 'logging']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate recognition config
        recognition = self.config['recognition']
        required_recognition = ['similarity_threshold', 'embedding_size']
        for field in required_recognition:
            if field not in recognition:
                raise ValueError(f"Missing recognition config field: {field}")
        
        # Validate tracking config
        tracking = self.config['tracking']
        required_tracking = ['max_age', 'min_hits', 'iou_threshold']
        for field in required_tracking:
            if field not in tracking:
                raise ValueError(f"Missing tracking config field: {field}")
        
        logger.debug("Configuration validation passed")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def update(self, key: str, value: Any):
        """Update configuration value by dot notation key"""
        keys = key.split('.')
        target = self.config
        
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        target[keys[-1]] = value
        logger.debug(f"Updated config: {key} = {value}")
    
    def save(self, path: str = None):
        """Save current configuration to file"""
        save_path = path or self.config_path
        
        try:
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def get_detection_skip_frames(self) -> int:
        """Get number of frames to skip between detection cycles"""
        return self.config['detection'].get('frames_to_skip', 5)
    
    def get_confidence_threshold(self) -> float:
        """Get detection confidence threshold"""
        return self.config['detection']['confidence_threshold']
    
    def get_similarity_threshold(self) -> float:
        """Get recognition similarity threshold"""
        return self.config['recognition']['similarity_threshold']