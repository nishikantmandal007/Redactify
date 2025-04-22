#!/usr/bin/env python3
# Redactify/services/cleanup.py

import os
import time
import logging
from datetime import datetime, timedelta
import shutil

from ..core.config import (
    UPLOAD_DIR,
    TEMP_DIR,
    TEMP_FILE_MAX_AGE_SECONDS
)

# Configure logging
logger = logging.getLogger(__name__)

def cleanup_temp_files(force=False):
    """
    Clean up temporary and uploaded files.
    If force=True, removes all files regardless of age.
    Otherwise, removes files older than the configured maximum age.
    
    Args:
        force (bool): If True, remove all files. Defaults to False.
        
    Returns:
        int: Number of files removed
    """
    if not (TEMP_DIR and os.path.isdir(TEMP_DIR)):
        logger.warning(f"Temp directory not found: {TEMP_DIR}")
        return 0
        
    if not (UPLOAD_DIR and os.path.isdir(UPLOAD_DIR)):
        logger.warning(f"Upload directory not found: {UPLOAD_DIR}")
        return 0
        
    # Current timestamp and cutoff logic (only used if force=False)
    now = time.time()
    max_age_seconds = TEMP_FILE_MAX_AGE_SECONDS
    cutoff_time = now - max_age_seconds
    cutoff_datetime = datetime.fromtimestamp(cutoff_time)
    formatted_cutoff = cutoff_datetime.strftime('%Y-%m-%d %H:%M:%S')
    
    if force:
        logger.info("Performing forced cleanup - removing ALL files from temp/upload directories.")
    else:
        logger.info(f"Cleaning up files older than {formatted_cutoff} ({max_age_seconds} seconds)")
    
    removed_count = 0
    
    # Process both directories
    for directory in [TEMP_DIR, UPLOAD_DIR]:
        logger.info(f"Processing directory: {directory}")
        
        try:
            # List all files in the directory
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                
                # Skip directories and symlinks for safety
                if os.path.isdir(file_path) or os.path.islink(file_path):
                    logger.debug(f"Skipping directory or symlink: {file_path}")
                    continue
                    
                # Check age only if force is False
                should_remove = False
                if force:
                    should_remove = True
                else:
                    # Get file's last modification time
                    try:
                        file_mod_time = os.path.getmtime(file_path)
                        if file_mod_time < cutoff_time:
                            should_remove = True
                    except FileNotFoundError:
                        logger.warning(f"File not found during cleanup check: {file_path}")
                        continue # Skip if file disappeared
                    except Exception as time_err:
                        logger.error(f"Error getting modification time for {file_path}: {time_err}")
                        continue # Skip if cannot get time
                
                # Remove the file if needed
                if should_remove:
                    try:
                        # Remove the file
                        os.remove(file_path)
                        reason = "Forced cleanup" if force else "Old file"
                        logger.info(f"Removed ({reason}): {file_path}")
                        removed_count += 1
                    except Exception as e:
                        logger.error(f"Error removing file {file_path}: {e}", exc_info=True)
                        
        except Exception as e:
            logger.error(f"Error processing directory {directory}: {e}", exc_info=True)
            
    logger.info(f"Cleanup completed. Removed {removed_count} files.")
    return removed_count
    
def delete_user_files(task_id):
    """
    Delete all files associated with a specific task ID.
    
    Args:
        task_id: The task ID associated with the files to delete
    
    Returns:
        int: Number of files removed
    """
    if not task_id:
        logger.warning("No task ID provided for file deletion")
        return 0
        
    removed_count = 0
    
    # Process both directories
    for directory in [TEMP_DIR, UPLOAD_DIR]:
        if not os.path.isdir(directory):
            continue
            
        try:
            # List all files in the directory
            for filename in os.listdir(directory):
                # Check if filename contains the task ID
                if task_id in filename:
                    file_path = os.path.join(directory, filename)
                    
                    # Safety check - ensure it's a file
                    if os.path.isfile(file_path) and not os.path.islink(file_path):
                        try:
                            # Remove the file
                            os.remove(file_path)
                            logger.info(f"Removed user file: {file_path}")
                            removed_count += 1
                        except Exception as e:
                            logger.error(f"Error removing file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
            
    return removed_count