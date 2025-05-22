#!/usr/bin/env python3
# Redactify/core/analyzers.py

import logging
import paddleocr
import os
import threading
import gc
import psutil
import time
from ..recognizers import custom_recognizer_list, get_custom_pii_entity_names
# Import new config for TF NLP memory fraction
from .config import PRESIDIO_CONFIG as PRESDIO_CONFIG, GPU_MEMORY_FRACTION_TF_NLP 
from .pii_types import get_pii_types, get_pii_friendly_names, get_all_pii_types
from ..utils.gpu_utils import is_gpu_available, initialize_paddle_gpu, configure_gpu_memory, get_gpu_info

# --- Presidio Setup (Integrates Custom Recognizers) ---
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.recognizer_registry import RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    
    # Configure GPU before initializing NLP engine
    if is_gpu_available():
        # Get GPU info for logging
        gpu_info = get_gpu_info()
        logging.info(f"Initializing NLP engine with GPU acceleration: {gpu_info}")
        
        # Configure GPU memory specifically for the Presidio/Spacy TensorFlow backend.
        # This helps reserve a portion of GPU memory for NLP tasks, leaving the rest for PaddleOCR.
        configure_gpu_memory(memory_fraction=GPU_MEMORY_FRACTION_TF_NLP)
        logging.info(f"NLP TF GPU memory fraction set to: {GPU_MEMORY_FRACTION_TF_NLP}")
        
        # Set TensorFlow environment variables for better GPU performance
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        # Set environment variable for better PyTorch GPU performance (if used)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    else:
        logging.info("No GPU detected, using CPU for NLP engine")
    
    # Create NLP engine with appropriate configuration
    provider = NlpEngineProvider(nlp_configuration=PRESDIO_CONFIG.get('nlp_config', {}))
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(languages=PRESDIO_CONFIG.get('supported_languages', ["en"]))
    
    if custom_recognizer_list:
        for recognizer in custom_recognizer_list:
            registry.add_recognizer(recognizer)  # Add custom ones
            
    analyzer = AnalyzerEngine(
        registry=registry, 
        nlp_engine=nlp_engine, 
        supported_languages=PRESDIO_CONFIG.get('supported_languages', ["en"])
    )
    logging.info("Presidio Analyzer initialized with custom recognizers and GPU acceleration if available")
except Exception as e:
    analyzer = None
    logging.error(f"Presidio initialization failed: {e}", exc_info=True)

# --- PaddleOCR Init with GPU Support ---
paddle_init_lock = threading.Lock()
ocr = None

def get_paddle_ocr():
    """Get or initialize PaddleOCR with GPU acceleration if available."""
    global ocr
    
    # Return existing instance if already initialized
    if ocr is not None:
        return ocr
        
    # Thread-safe initialization
    with paddle_init_lock:
        # Check again in case another thread initialized while waiting
        if ocr is not None:
            return ocr
            
        try:
            # Initialize PaddleOCR with GPU support.
            # PaddleOCR manages its GPU memory dynamically. By pre-allocating memory for TensorFlow
            # (via GPU_MEMORY_FRACTION_TF_GENERAL and GPU_MEMORY_FRACTION_TF_NLP in config),
            # we ensure TensorFlow doesn't consume all GPU memory, leaving room for PaddleOCR.
            # The effectiveness depends on the sum of TF fractions being less than 1.0,
            # ideally < 0.8 to also account for system overhead and PaddleOCR's needs.
            gpu_enabled = initialize_paddle_gpu()
            
            # Configure OCR parameters based on available resources
            # These significantly affect OCR performance and memory usage
            use_angle_cls = not psutil.virtual_memory().percent > 80  # Disable angle classifier if memory is tight
            
            # Create PaddleOCR instance with appropriate device
            paddle_args = {
                'use_angle_cls': use_angle_cls,
                'lang': 'en',
                'show_log': False,
                'use_gpu': gpu_enabled,
                'enable_mkldnn': not gpu_enabled  # Use MKL-DNN acceleration on CPU
            }
            
            # Add GPU-specific optimizations
            if gpu_enabled:
                # Check available GPU memory and adjust batch size accordingly
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    free_memory_gb = info.free / (1024**3)
                    
                    # Adjust OCR parameters based on available GPU memory
                    if free_memory_gb > 4:  # More than 4GB available
                        paddle_args['use_tensorrt'] = True  # Enable TensorRT acceleration
                        paddle_args['precision'] = 'fp16'   # Use mixed precision
                    elif free_memory_gb < 2:  # Less than 2GB available
                        # Lower batch size and limit threads for constrained environments
                        paddle_args['max_batch_size'] = 8
                        paddle_args['max_text_length'] = 40
                    
                    pynvml.nvmlShutdown()
                except (ImportError, Exception) as e:
                    logging.warning(f"Could not query GPU memory details: {e}")
            
            ocr = paddleocr.PaddleOCR(**paddle_args)
            
            device_type = "GPU" if gpu_enabled else "CPU"
            logging.info(f"PaddleOCR initialized successfully using {device_type} with parameters: {paddle_args}")
            return ocr
        except Exception as e:
            logging.error(f"PaddleOCR initialization failed: {e}", exc_info=True)
            return None

# Initialize PaddleOCR at module load time
try:
    ocr = get_paddle_ocr()
except Exception as e:
    logging.error(f"PaddleOCR initialization error: {e}", exc_info=True)

# --- PII Types Initialization ---
# For backward compatibility
COMMON_PII_TYPES = get_pii_friendly_names("common")
ADVANCED_PII_TYPES = get_pii_friendly_names("advanced")
PII_FRIENDLY_NAMES = get_all_pii_types()

class AnalyzerFactory:
    """Factory class to create and configure PII analyzers with GPU acceleration."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Create singleton instance to avoid redundant analyzer initialization."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AnalyzerFactory, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize factory with required components."""
        if self._initialized:
            return
            
        self._initialized = True
        self._analyzer = analyzer
        self._ocr_engine = ocr
        self._gpu_available = is_gpu_available()
        
        # Log initialization with GPU status
        device_type = "GPU enabled" if self._gpu_available else "CPU only"
        logging.info(f"AnalyzerFactory initialized ({device_type})")
    
    @staticmethod
    def get_analyzer(pii_types=None):
        """
        Get properly configured analyzer instance for the requested PII types.
        
        Args:
            pii_types: List of PII types to analyze for
            
        Returns:
            AnalyzerEngine: Configured analyzer instance
        """
        if not analyzer:
            logging.error("Analyzer not available - initialization failed")
            return None
        
        # Check memory before returning
        if AnalyzerFactory.should_clean_memory():
            AnalyzerFactory.cleanup_resources()
            
        return analyzer
    
    @staticmethod
    def get_ocr_engine(force_reinit=False):
        """
        Get OCR engine instance with GPU acceleration when available.
        
        Args:
            force_reinit: Force reinitialization of OCR engine (e.g., after memory issues)
            
        Returns:
            PaddleOCR: OCR engine instance
        """
        global ocr
        
        # Force reinitialization if requested
        if force_reinit and ocr is not None:
            logging.info("Forcing OCR engine reinitialization")
            with paddle_init_lock:
                ocr = None
                gc.collect()  # Force garbage collection
                
        # Check memory before returning OCR engine
        if AnalyzerFactory.should_clean_memory():
            AnalyzerFactory.cleanup_resources()
                
        return get_paddle_ocr()
    
    @staticmethod
    def should_clean_memory(threshold=85):
        """Check if memory usage is high enough to warrant cleanup."""
        memory_usage = psutil.virtual_memory().percent
        return memory_usage > threshold
    
    @staticmethod
    def cleanup_resources():
        """Perform memory cleanup when resources get tight."""
        logging.info("High memory usage detected, performing cleanup")
        
        # Collect garbage first
        collected = gc.collect(2)  # Full collection
        logging.info(f"Garbage collection freed {collected} objects")
        
        # Sleep briefly to allow OS to reclaim memory
        time.sleep(0.1)
        
    @staticmethod
    def is_gpu_mode():
        """Check if running in GPU-accelerated mode."""
        return is_gpu_available()
    
    @staticmethod
    def get_device_info():
        """Get information about the current processing device (CPU/GPU)."""
        if is_gpu_available():
            try:
                return get_gpu_info()
            except Exception as e:
                logging.warning(f"Error getting GPU info: {e}")
                return "GPU available (details unknown)"
        else:
            # Get CPU info
            try:
                import multiprocessing
                cpu_count = multiprocessing.cpu_count()
                return f"CPU only (cores: {cpu_count})"
            except:
                return "CPU only"