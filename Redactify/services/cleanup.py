#!/usr/bin/env python3
# Redactify/services/cleanup.py

import os
import time
import logging
from ..core.config import TEMP_DIR, UPLOAD_DIR

def cleanup_temp_files(max_age_seconds=None, force=False):
    """
    Removes temporary files older than specified age.
    
    Args:
        max_age_seconds: Maximum age of files to keep (in seconds)
        force: If True, remove all files except .gitkeep regardless of age
        
    Returns:
        int: Number of files removed
    """
    if max_age_seconds is None:
        # Default to 1 hour if not specified
        max_age_seconds = 3600
        
    try:
        now = time.time()
        count = 0
        
        for dirname in [TEMP_DIR, UPLOAD_DIR]:
            if not os.path.exists(dirname):
                logging.warning(f"Directory does not exist: {dirname}")
                continue
                
            for filename in os.listdir(dirname):
                # Skip .gitkeep files which are used to ensure the directory exists in git
                if filename == '.gitkeep':
                    continue
                    
                file_path = os.path.join(dirname, filename)
                
                # Check if it's a file (not directory)
                if os.path.isfile(file_path):
                    # If force is True, remove all files regardless of age
                    # Otherwise, check file age
                    file_age = now - os.path.getmtime(file_path)
                    
                    if force or file_age > max_age_seconds:
                        try:
                            os.remove(file_path)
                            count += 1
                            logging.debug(f"Removed file: {file_path} (Age: {file_age:.1f}s)")
                        except Exception as e:
                            logging.error(f"Failed to remove {file_path}: {e}")
        
        if count > 0:
            logging.info(f"Cleanup removed {count} files")
            
        return count
            
    except Exception as e:
        logging.error(f"Error during file cleanup: {e}", exc_info=True)
        return 0