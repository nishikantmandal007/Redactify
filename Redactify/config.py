# Redactify/config.py - Revised structure with Confidence Threshold
import os
import yaml
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - CONFIG - %(message)s') # Added CONFIG tag
load_dotenv()

# --- Determine Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Project root is parent of Redactify/
config_path = os.path.join(project_root, 'config.yaml') # Look for config.yaml in root

# --- Default Configuration ---
DEFAULT_CONFIG = {
    "presidio_config": { # Define default Presidio structure
        "nlp_config": {"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]},
        "supported_languages": ["en"]
        },
    "temp_dir": "temp_files",
    "upload_dir": "upload_files",
    "redis_url": "redis://localhost:6379/0",
    "max_file_size_mb": 50,
    "ocr_confidence_threshold": 0.7, # Default OCR confidence
    "presidio_confidence_threshold": 0.35, # NEW: Default Presidio confidence
    "celery_task_soft_time_limit": 300,
    "celery_task_hard_time_limit": 360,
    "temp_file_max_age_seconds": 86400 # Default 1 day
}
# Define required keys based on defaults (ensures structure check)
REQUIRED_KEYS = DEFAULT_CONFIG.keys()

# --- Load User Configuration ---
config = {} # Start with empty config
user_config = {} # Initialize user_config
try:
    logging.info(f"Attempting to load configuration from: {config_path}")
    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f) or {} # Load user config, ensure it's a dict even if file is empty
    logging.info(f"Successfully loaded user configuration from {config_path}")
except FileNotFoundError:
    logging.warning(f"Configuration file not found at {config_path}. Using default settings.")
    # No need to load user_config, it remains empty
except yaml.YAMLError as e:
    logging.error(f"Error parsing configuration file {config_path}: {e}. Using default settings.")
    # Use default, user_config remains empty
except Exception as e:
    logging.error(f"Could not load configuration file {config_path}: {e}. Using default settings.")
    # Use default, user_config remains empty

# Merge defaults with user config (user settings override defaults)
# Need a slightly more careful merge for nested dicts like presidio_config
config = DEFAULT_CONFIG.copy() # Start with a copy of defaults
for key, value in user_config.items():
    if isinstance(value, dict) and isinstance(config.get(key), dict):
        # Merge nested dictionaries (like presidio_config)
        config[key] = {**config.get(key, {}), **value}
    else:
        # Overwrite top-level keys
        config[key] = value

# Basic Validation: Check if all *originally required* keys are present after merge
# (though defaults ensure they are, this could catch issues if defaults change)
# final_keys = config.keys()
# missing_keys = [key for key in REQUIRED_KEYS if key not in final_keys]
# if missing_keys:
#     # This condition should ideally not be met if defaults are handled correctly
#     logging.error(f"Critical config keys missing after merge: {', '.join(missing_keys)}. Check default settings.")
#     exit(1) # Exit if fundamental keys are somehow missing

# --- Define Exported Variables (using merged config) ---
# Define paths first using the final config values
UPLOAD_DIR_NAME = config.get("upload_dir")
TEMP_DIR_NAME = config.get("temp_dir")
UPLOAD_DIR = os.path.abspath(os.path.join(project_root, UPLOAD_DIR_NAME))
TEMP_DIR = os.path.abspath(os.path.join(project_root, TEMP_DIR_NAME))

# Define other variables
PRESDIO_CONFIG = config.get("presidio_config")
REDIS_URL = config.get("redis_url")
# Use try-except for type casting to handle potential non-numeric values in user yaml
try:
    MAX_FILE_SIZE_MB = int(config.get("max_file_size_mb"))
except (ValueError, TypeError):
    logging.warning(f"Invalid value for max_file_size_mb in config. Using default: {DEFAULT_CONFIG['max_file_size_mb']}")
    MAX_FILE_SIZE_MB = DEFAULT_CONFIG['max_file_size_mb']

try:
    OCR_CONFIDENCE_THRESHOLD = float(config.get("ocr_confidence_threshold"))
except (ValueError, TypeError):
    logging.warning(f"Invalid value for ocr_confidence_threshold in config. Using default: {DEFAULT_CONFIG['ocr_confidence_threshold']}")
    OCR_CONFIDENCE_THRESHOLD = DEFAULT_CONFIG['ocr_confidence_threshold']

try:
    PRESIDIO_CONFIDENCE_THRESHOLD = float(config.get("presidio_confidence_threshold")) # Load the new setting
except (ValueError, TypeError):
    logging.warning(f"Invalid value for presidio_confidence_threshold in config. Using default: {DEFAULT_CONFIG['presidio_confidence_threshold']}")
    PRESIDIO_CONFIDENCE_THRESHOLD = DEFAULT_CONFIG['presidio_confidence_threshold']

try:
    CELERY_TASK_SOFT_TIME_LIMIT = int(config.get("celery_task_soft_time_limit"))
except (ValueError, TypeError):
    logging.warning(f"Invalid value for celery_task_soft_time_limit in config. Using default: {DEFAULT_CONFIG['celery_task_soft_time_limit']}")
    CELERY_TASK_SOFT_TIME_LIMIT = DEFAULT_CONFIG['celery_task_soft_time_limit']

try:
    CELERY_TASK_HARD_TIME_LIMIT = int(config.get("celery_task_hard_time_limit"))
except (ValueError, TypeError):
    logging.warning(f"Invalid value for celery_task_hard_time_limit in config. Using default: {DEFAULT_CONFIG['celery_task_hard_time_limit']}")
    CELERY_TASK_HARD_TIME_LIMIT = DEFAULT_CONFIG['celery_task_hard_time_limit']

try:
    TEMP_FILE_MAX_AGE_SECONDS = int(config.get("temp_file_max_age_seconds"))
except (ValueError, TypeError):
    logging.warning(f"Invalid value for temp_file_max_age_seconds in config. Using default: {DEFAULT_CONFIG['temp_file_max_age_seconds']}")
    TEMP_FILE_MAX_AGE_SECONDS = DEFAULT_CONFIG['temp_file_max_age_seconds']

MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
# --- End Exported Variables ---


# --- Create Directories ---
# Create directories *after* paths are fully defined and validated
try:
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    logging.info(f"Upload directory set to: {UPLOAD_DIR}")
    logging.info(f"Temporary directory set to: {TEMP_DIR}")
except OSError as e:
    logging.error(f"CRITICAL: Error creating essential directories ({TEMP_DIR}, {UPLOAD_DIR}): {e}")
    exit(1) # Exit if essential directories cannot be created
# --- End Create Directories ---

# --- Log Final Config Values (optional for debugging) ---
logging.debug(f"Final configuration loaded: {config}")
logging.info(f"Using Presidio confidence threshold: {PRESIDIO_CONFIDENCE_THRESHOLD}")
logging.info(f"Using OCR confidence threshold: {OCR_CONFIDENCE_THRESHOLD}")