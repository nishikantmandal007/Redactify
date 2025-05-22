#!/usr/bin/env python3
# Redactify/core/config.py - Configuration management module
import os
import yaml
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - CONFIG - %(message)s')
# Load environment variables from .env file
load_dotenv()

# --- Determine Paths ---
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Adjusted for new location
project_root = os.path.dirname(script_dir)  # Project root is parent of Redactify/
config_path = os.path.join(project_root, 'config.yaml')  # Look for config.yaml in root

# --- Default Configuration ---
DEFAULT_CONFIG = {
    "presidio_config": {  # Define default Presidio structure
        "nlp_config": {"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]},
        "supported_languages": ["en"]
    },
    "temp_dir": "temp_files",
    "upload_dir": "upload_files",
    "redis_url": "redis://localhost:6379/0",
    "max_file_size_mb": 50,
    "ocr_confidence_threshold": 0.7,  # Default OCR confidence
    "presidio_confidence_threshold": 0.35,  # Default Presidio confidence
    "celery_task_soft_time_limit": 300,
    "celery_task_hard_time_limit": 360,
    "temp_file_max_age_seconds": 86400,  # Default 1 day
    "host": "127.0.0.1",
    "port": 5000,
    "log_level": "INFO",
    "task_max_memory_percent": 85,  # Default max memory percent for tasks
    "task_healthy_cpu_percent": 80,  # Default healthy CPU percent for tasks
    "gpu_memory_fraction_tf_general": 0.2, # Default TensorFlow general GPU memory fraction
    "gpu_memory_fraction_tf_nlp": 0.3      # Default TensorFlow NLP-specific GPU memory fraction
}

# Define required keys based on defaults (ensures structure check)
REQUIRED_KEYS = DEFAULT_CONFIG.keys()

# --- Environment Variable Mapping ---
# Map environment variables to config keys (with REDACTIFY_ prefix)
ENV_VAR_MAPPING = {
    "REDACTIFY_HOST": "host",
    "REDACTIFY_PORT": "port",
    "REDACTIFY_REDIS_URL": "redis_url",
    "REDACTIFY_MAX_FILE_SIZE_MB": "max_file_size_mb",
    "REDACTIFY_OCR_THRESHOLD": "ocr_confidence_threshold",
    "REDACTIFY_PRESIDIO_THRESHOLD": "presidio_confidence_threshold",
    "REDACTIFY_TEMP_FILE_AGE": "temp_file_max_age_seconds",
    "REDACTIFY_LOG_LEVEL": "log_level",
    "REDACTIFY_TASK_MAX_MEMORY_PERCENT": "task_max_memory_percent",
    "REDACTIFY_TASK_HEALTHY_CPU_PERCENT": "task_healthy_cpu_percent",
    "REDACTIFY_GPU_MEMORY_FRACTION_TF_GENERAL": "gpu_memory_fraction_tf_general",
    "REDACTIFY_GPU_MEMORY_FRACTION_TF_NLP": "gpu_memory_fraction_tf_nlp",
}


def load_config():
    """Load and merge configuration from file with defaults and environment variables"""
    # --- Load User Configuration from YAML ---
    user_config = {}  # Initialize user_config
    try:
        logging.info(f"Attempting to load configuration from: {config_path}")
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}  # Load user config, ensure it's a dict even if file is empty
        logging.info(f"Successfully loaded user configuration from {config_path}")
    except FileNotFoundError:
        logging.warning(f"Configuration file not found at {config_path}. Using default settings.")
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {config_path}: {e}. Using default settings.")
    except Exception as e:
        logging.error(f"Could not load configuration file {config_path}: {e}. Using default settings.")

    # Merge defaults with user config (user settings override defaults)
    # Need a slightly more careful merge for nested dicts like presidio_config
    config = DEFAULT_CONFIG.copy()  # Start with a copy of defaults
    for key, value in user_config.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            # Merge nested dictionaries (like presidio_config)
            config[key] = {**config.get(key, {}), **value}
        else:
            # Overwrite top-level keys
            config[key] = value
            
    # --- Override with Environment Variables ---
    # Environment variables take precedence over config.yaml
    for env_var, config_key in ENV_VAR_MAPPING.items():
        if env_var in os.environ:
            env_value = os.environ.get(env_var)
            original_type = type(config[config_key])
            
            # Attempt to convert the environment variable to the same type as the config value
            try:
                if original_type == bool:
                    # Special handling for boolean values
                    if env_value.lower() in ('true', 't', 'yes', 'y', '1'):
                        config[config_key] = True
                    elif env_value.lower() in ('false', 'f', 'no', 'n', '0'):
                        config[config_key] = False
                elif original_type == int:
                    config[config_key] = int(env_value)
                elif original_type == float:
                    config[config_key] = float(env_value)
                else:
                    config[config_key] = env_value
                    
                logging.info(f"Overriding config '{config_key}' with environment variable {env_var}")
            except (ValueError, TypeError) as e:
                logging.warning(f"Could not convert environment variable {env_var}='{env_value}' to type {original_type}: {e}")

    return config


# Load configuration
config = load_config()

# --- Define Exported Variables (using merged config) ---
# Define paths first using the final config values
UPLOAD_DIR_NAME = config.get("upload_dir")
TEMP_DIR_NAME = config.get("temp_dir")
UPLOAD_DIR = os.path.abspath(os.path.join(project_root, UPLOAD_DIR_NAME))
TEMP_DIR = os.path.abspath(os.path.join(project_root, TEMP_DIR_NAME))

# Define other variables
PRESIDIO_CONFIG = config.get("presidio_config")
REDIS_URL = config.get("redis_url")
HOST = config.get("host")
PORT = config.get("port")

def set_config_paths(upload_dir=None, temp_dir=None):
    """Override the configuration paths for testing or other purposes."""
    global UPLOAD_DIR, TEMP_DIR
    if upload_dir is not None:
        UPLOAD_DIR = upload_dir
    if temp_dir is not None:
        TEMP_DIR = temp_dir
    logging.info(f"Configuration paths overridden - Upload dir: {UPLOAD_DIR}, Temp dir: {TEMP_DIR}")

def get_config_value(key, default_key=None, value_type=None):
    """Helper to safely get a config value with type conversion"""
    try:
        value = config.get(key)
        if value_type is not None and value is not None:
            return value_type(value)
        return value
    except (ValueError, TypeError):
        default_value = DEFAULT_CONFIG[default_key or key]
        logging.warning(f"Invalid value for {key} in config. Using default: {default_value}")
        return default_value


# Use the helper function for safer type conversion
MAX_FILE_SIZE_MB = get_config_value("max_file_size_mb", value_type=int)
OCR_CONFIDENCE_THRESHOLD = get_config_value("ocr_confidence_threshold", value_type=float)
PRESIDIO_CONFIDENCE_THRESHOLD = get_config_value("presidio_confidence_threshold", value_type=float)
CELERY_TASK_SOFT_TIME_LIMIT = get_config_value("celery_task_soft_time_limit", value_type=int)
CELERY_TASK_HARD_TIME_LIMIT = get_config_value("celery_task_hard_time_limit", value_type=int)
TEMP_FILE_MAX_AGE_SECONDS = get_config_value("temp_file_max_age_seconds", value_type=int)
LOG_LEVEL = get_config_value("log_level")

# Task resource limits
TASK_MAX_MEMORY_PERCENT = get_config_value("task_max_memory_percent", value_type=int)
"""Maximum memory percentage a task should consume before triggering warnings or retries."""

TASK_HEALTHY_CPU_PERCENT = get_config_value("task_healthy_cpu_percent", value_type=int)
"""CPU percentage considered healthy; tasks might be skipped or rescheduled above this."""

# GPU Memory Configuration Strategy:
# TensorFlow (used by Spacy/Presidio) benefits from having its GPU memory fraction explicitly set
# via `set_logical_device_configuration`. This helps prevent TF from allocating all available GPU memory.
# PaddleOCR (used for OCR tasks) typically tries to allocate the memory it needs dynamically.
# By setting TF's memory limits, we reserve memory for TF and leave the rest for PaddleOCR and system overhead.
# The sum of GPU_MEMORY_FRACTION_TF_GENERAL and GPU_MEMORY_FRACTION_TF_NLP should ideally be < 0.8
# to leave ample room for PaddleOCR, other potential GPU users, and system overhead.
# Users might need to adjust these fractions based on their specific GPU memory capacity and workload.
GPU_MEMORY_FRACTION_TF_GENERAL = get_config_value("gpu_memory_fraction_tf_general", value_type=float)
"""GPU memory fraction for general TensorFlow operations (e.g., in app.py initializations)."""

GPU_MEMORY_FRACTION_TF_NLP = get_config_value("gpu_memory_fraction_tf_nlp", value_type=float)
"""GPU memory fraction specifically for the Presidio/Spacy TensorFlow backend (in analyzers.py)."""

MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# --- Create Directories ---
try:
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    logging.info(f"Upload directory set to: {UPLOAD_DIR}")
    logging.info(f"Temporary directory set to: {TEMP_DIR}")
except OSError as e:
    logging.error(f"CRITICAL: Error creating essential directories ({TEMP_DIR}, {UPLOAD_DIR}): {e}")
    exit(1)  # Exit if essential directories cannot be created

# --- Set Logging Level ---
logging_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.getLogger().setLevel(logging_level)

# --- Log Final Config Values ---
logging.debug(f"Final configuration loaded: {config}")
logging.info(f"Using Presidio confidence threshold: {PRESIDIO_CONFIDENCE_THRESHOLD}")
logging.info(f"Using OCR confidence threshold: {OCR_CONFIDENCE_THRESHOLD}")
logging.info(f"Redis URL: {REDIS_URL}")
logging.info(f"Max file size: {MAX_FILE_SIZE_MB} MB")
logging.info(f"Log level: {LOG_LEVEL}")
logging.info(f"Task max memory percent: {TASK_MAX_MEMORY_PERCENT}%")
logging.info(f"Task healthy CPU percent: {TASK_HEALTHY_CPU_PERCENT}%")
logging.info(f"GPU Memory Fraction for TensorFlow (General): {GPU_MEMORY_FRACTION_TF_GENERAL}")
logging.info(f"GPU Memory Fraction for TensorFlow (NLP): {GPU_MEMORY_FRACTION_TF_NLP}")