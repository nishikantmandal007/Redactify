#!/usr/bin/env python3
# Redactify/services/cleanup.py

import os
import time
import logging
from ..core.config import TEMP_DIR

def cleanup_temp_files(max_age_seconds=None):
    """
    Removes temporary files older than specified age.
    
    Args:
        max_age_seconds: Maximum age of files to keep (in seconds)
        
    Returns:
        int: Number of files removed
    """
    if max_age_seconds is None:
        # Default to 1 hour if not specified
        max_age_seconds = 3600
        
    try:
        now = time.time()
        count = 0
        
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            
            # Check if it's a file (not directory) and older than max age
            if os.path.isfile(file_path):
                file_age = now - os.path.getmtime(file_path)
                
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    count += 1
                    logging.debug(f"Removed temp file: {filename} (Age: {file_age:.1f}s)")
        
        if count > 0:
            logging.info(f"Cleanup removed {count} temporary files")
            
        return count
            
    except Exception as e:
        logging.error(f"Error during temp file cleanup: {e}", exc_info=True)
        return 0