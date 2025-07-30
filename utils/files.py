"""
Shared utility functions for file operations across the cupid package.

This module provides common functionality for:
- JSON file loading and saving with consistent formatting
- Directory creation and management
- Error handling for file operations
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

def load_json(path: str, default: Any = None) -> Any:
    """
    Load JSON data from file if it exists, otherwise return default value.
    
    Args:
        path: Path to the JSON file
        default: Default value to return if file doesn't exist or can't be loaded
        
    Returns:
        Loaded JSON data or default value
    """
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load JSON from {path}: {e}")
            return default
    return default

def save_json(path: str, data: Any, indent: int = 4) -> bool:
    """
    Save data to JSON file with pretty formatting.
    
    Args:
        path: Path to save the JSON file
        data: Data to save
        indent: Number of spaces for indentation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(data, f, indent=indent)
        return True
    except (IOError, TypeError) as e:
        logger.error(f"Failed to save JSON to {path}: {e}")
        return False

def ensure_directory(path: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
        
    Returns:
        True if directory exists or was created successfully, False otherwise
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False 