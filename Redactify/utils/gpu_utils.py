#!/usr/bin/env python3
# Redactify/utils/gpu_utils.py

import logging
import os
import sys
import subprocess
import numpy as np
import gc
import time
from typing import Optional, Union, Dict, Any, List, Tuple
import cv2
import tensorflow as tf

# Global variables to cache detection results
_GPU_AVAILABLE = None
_GPU_INFO = None
_CUDA_OPENCV = None
_PADDLE_GPU_INITIALIZED = False

def is_gpu_available() -> bool:
    """
    Check if GPU is available for computation.
    Caches result to avoid repeated checks.
    
    Returns:
        bool: True if a compatible GPU is available, False otherwise
    """
    global _GPU_AVAILABLE
    
    # Return cached result if already checked
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE
    
    try:
        # First, check if CUDA is available through specific libraries
        try:
            # Check through OpenCV CUDA
            import cv2
            cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_device_count > 0:
                _GPU_AVAILABLE = True
                logging.info(f"GPU available: {cuda_device_count} CUDA device(s) found through OpenCV")
                return True
        except (ImportError, AttributeError, cv2.error):
            pass  # OpenCV CUDA not available
            
        # Check if NVIDIA SMI is available (more reliable for NVIDIA GPUs)
        nvidia_smi_output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if nvidia_smi_output.returncode == 0 and "name" in nvidia_smi_output.stdout:
            _GPU_AVAILABLE = True
            logging.info("GPU available: NVIDIA GPU detected through nvidia-smi")
            return True
            
        # Check through environment variable (e.g., set by container runtime)
        if os.environ.get("NVIDIA_VISIBLE_DEVICES", "") not in ["", "none", "void"]:
            _GPU_AVAILABLE = True
            logging.info("GPU available: NVIDIA_VISIBLE_DEVICES environment variable is set")
            return True
            
        # If we reach here, no GPU was detected
        _GPU_AVAILABLE = False
        logging.info("No GPU detected, using CPU for computation")
        return False
        
    except Exception as e:
        logging.warning(f"Error checking GPU availability: {e}")
        _GPU_AVAILABLE = False
        return False

def get_gpu_info() -> Dict[str, Any]:
    """
    Get detailed information about available GPUs.
    
    Returns:
        Dict[str, Any]: Dictionary containing GPU information or empty dict if no GPU
    """
    global _GPU_INFO
    
    # Return cached info if already collected
    if _GPU_INFO is not None:
        return _GPU_INFO
    
    gpu_info = {
        "available": False,
        "count": 0,
        "devices": [],
        "memory_total_mb": 0,
        "memory_free_mb": 0,
        "driver_version": ""
    }
    
    try:
        # Only proceed if GPU is available
        if not is_gpu_available():
            _GPU_INFO = gpu_info
            return gpu_info
            
        # Try to get detailed info via nvidia-smi for NVIDIA GPUs
        nvidia_smi_output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if nvidia_smi_output.returncode == 0:
            lines = nvidia_smi_output.stdout.strip().split('\n')
            gpu_count = len(lines)
            
            if gpu_count > 0:
                gpu_info["available"] = True
                gpu_info["count"] = gpu_count
                
                for line in lines:
                    if ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 4:
                            name = parts[0].strip()
                            memory_total = parts[1].strip()
                            memory_free = parts[2].strip()
                            driver = parts[3].strip()
                            
                            # Extract numeric values from memory string (e.g., "16384 MiB")
                            memory_total_mb = 0
                            memory_free_mb = 0
                            try:
                                total_parts = memory_total.split()
                                free_parts = memory_free.split()
                                if len(total_parts) == 2:
                                    memory_total_mb = int(float(total_parts[0]))
                                if len(free_parts) == 2:
                                    memory_free_mb = int(float(free_parts[0]))
                            except (ValueError, IndexError):
                                pass
                                
                            gpu_info["devices"].append({
                                "name": name,
                                "memory_total_mb": memory_total_mb,
                                "memory_free_mb": memory_free_mb,
                                "driver": driver
                            })
                            
                            # Track total memory
                            gpu_info["memory_total_mb"] += memory_total_mb
                            gpu_info["memory_free_mb"] += memory_free_mb
                            
                            # Use the first device's driver version
                            if not gpu_info["driver_version"] and driver:
                                gpu_info["driver_version"] = driver
        
        # Also check OpenCV CUDA capabilities
        try:
            import cv2
            cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
            gpu_info["opencv_cuda"] = cuda_device_count > 0
            
            # Get compute capability if available
            if cuda_device_count > 0:
                try:
                    import pycuda.driver as cuda
                    cuda.init()
                    device = cuda.Device(0)
                    compute_capability = device.compute_capability()
                    gpu_info["compute_capability"] = f"{compute_capability[0]}.{compute_capability[1]}"
                except ImportError:
                    pass
        except (ImportError, AttributeError):
            gpu_info["opencv_cuda"] = False
        
        # Try to get more detailed info using PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["pytorch_available"] = True
                gpu_info["pytorch_device_count"] = torch.cuda.device_count()
                gpu_info["pytorch_device_name"] = torch.cuda.get_device_name(0)
                
                # Get current memory usage
                try:
                    gpu_info["pytorch_memory_allocated_mb"] = torch.cuda.memory_allocated(0) / (1024 * 1024)
                    gpu_info["pytorch_memory_reserved_mb"] = torch.cuda.memory_reserved(0) / (1024 * 1024)
                except:
                    pass
            else:
                gpu_info["pytorch_available"] = False
        except ImportError:
            gpu_info["pytorch_available"] = False
            
        # Check TensorFlow GPU status
        try:
            import tensorflow as tf
            gpu_info["tensorflow_available"] = tf.config.list_physical_devices('GPU') != []
        except ImportError:
            gpu_info["tensorflow_available"] = False
        
        # Update the global cache
        _GPU_INFO = gpu_info
        
        if gpu_info["available"]:
            logging.info(f"GPU(s) available: {gpu_info['count']} device(s), "
                        f"{gpu_info['memory_total_mb']} MB total memory, "
                        f"{gpu_info['memory_free_mb']} MB free memory, "
                        f"driver version {gpu_info['driver_version']}")
            
            # Include framework support info
            frameworks = []
            if gpu_info.get("opencv_cuda", False):
                frameworks.append("OpenCV-CUDA")
            if gpu_info.get("pytorch_available", False):
                frameworks.append("PyTorch")
            if gpu_info.get("tensorflow_available", False):
                frameworks.append("TensorFlow")
            if frameworks:
                logging.info(f"GPU-enabled frameworks: {', '.join(frameworks)}")
        
        return gpu_info
        
    except Exception as e:
        logging.warning(f"Error getting GPU info: {e}")
        _GPU_INFO = gpu_info
        return gpu_info

def get_gpu_enabled_opencv():
    """
    Get GPU-enabled OpenCV CUDA module if available.
    
    Returns:
        Optional[cv2.cuda]: OpenCV CUDA module or None if not available
    """
    global _CUDA_OPENCV
    
    # Return cached module if already initialized
    if _CUDA_OPENCV is not None:
        return _CUDA_OPENCV
    
    try:
        import cv2
        cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
        
        if cuda_device_count > 0:
            # Set the device to use (usually 0 for the first GPU)
            cv2.cuda.setDevice(0)
            _CUDA_OPENCV = cv2.cuda
            logging.info(f"OpenCV CUDA initialized with {cuda_device_count} device(s)")
            return _CUDA_OPENCV
        else:
            _CUDA_OPENCV = None
            return None
    except (ImportError, AttributeError, cv2.error) as e:
        logging.debug(f"OpenCV CUDA not available: {e}")
        _CUDA_OPENCV = None
        return None

def configure_gpu_memory(memory_fraction: float = 0.7) -> bool:
    """
    Configure GPU memory usage to avoid out-of-memory errors.
    
    Args:
        memory_fraction: Fraction of GPU memory to allocate (0.0 to 1.0)
        
    Returns:
        bool: True if configuration was successful, False otherwise
    """
    if not is_gpu_available():
        return False
    
    try:
        # Try to configure TensorFlow if being used
        try:
            import tensorflow as tf
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                # Set memory growth to avoid allocating all memory at once
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                
                # Optionally limit memory usage
                if 0.0 < memory_fraction < 1.0:
                    for device in physical_devices:
                        tf.config.set_logical_device_configuration(
                            device,
                            [tf.config.LogicalDeviceConfiguration(
                                memory_limit=int(get_gpu_info()['memory_total_mb'] * memory_fraction)
                            )]
                        )
                logging.info(f"TensorFlow GPU memory configured with growth=True, fraction={memory_fraction}")
                return True
        except (ImportError, AttributeError, ValueError) as e:
            logging.debug(f"Could not configure TensorFlow GPU: {e}")
        
        # Try to configure PyTorch if being used
        try:
            import torch
            if torch.cuda.is_available():
                # Set to use a percentage of available memory
                if 0.0 < memory_fraction < 1.0:
                    torch.cuda.set_per_process_memory_fraction(memory_fraction)
                    logging.info(f"PyTorch GPU memory fraction set to {memory_fraction}")
                return True
        except (ImportError, AttributeError) as e:
            logging.debug(f"Could not configure PyTorch GPU: {e}")
            
        # Try to configure PaddlePaddle if being used
        try:
            import paddle
            if paddle.device.is_compiled_with_cuda():
                # Set GPU memory fraction
                paddle.device.set_device('gpu')
                if 0.0 < memory_fraction < 1.0:
                    paddle.device.cuda.set_memory_fraction(memory_fraction)
                    logging.info(f"PaddlePaddle GPU memory fraction set to {memory_fraction}")
                return True
        except (ImportError, AttributeError) as e:
            logging.debug(f"Could not configure PaddlePaddle GPU: {e}")
            
        return False
        
    except Exception as e:
        logging.warning(f"Error configuring GPU memory: {e}")
        return False

def initialize_paddle_gpu():
    """
    Initialize PaddlePaddle with GPU support if available.
    This should be called once at application startup.
    
    Returns:
        bool: True if GPU was successfully enabled for PaddlePaddle
    """
    global _PADDLE_GPU_INITIALIZED
    
    if _PADDLE_GPU_INITIALIZED:
        return True
        
    if not is_gpu_available():
        return False
        
    try:
        import paddle
        
        if paddle.device.is_compiled_with_cuda():
            # Set device to GPU
            paddle.device.set_device('gpu')
            
            # Configure memory usage to avoid OOM
            # Get GPU info to determine memory fraction
            gpu_info = get_gpu_info()
            total_memory = gpu_info.get('memory_total_mb', 0)
            
            # Use a more conservative memory fraction for constrained GPUs
            memory_fraction = 0.8
            if total_memory > 0 and total_memory < 4000:  # Less than 4GB
                memory_fraction = 0.6  # Be more conservative
                
            paddle.device.cuda.set_memory_fraction(fraction=memory_fraction)
            
            # Test if GPU is actually working
            try:
                # Create a small tensor on GPU to test
                x = paddle.to_tensor([1.0, 2.0, 3.0], place=paddle.CUDAPlace(0))
                result = paddle.sum(x).numpy()
                
                _PADDLE_GPU_INITIALIZED = True
                logging.info(f"PaddlePaddle GPU acceleration successfully initialized (memory fraction: {memory_fraction})")
                return True
            except Exception as test_error:
                logging.warning(f"PaddlePaddle GPU test failed: {test_error}")
                # Fall back to CPU
                paddle.device.set_device('cpu')
                return False
        else:
            logging.info("PaddlePaddle not compiled with CUDA support, using CPU")
            return False
            
    except ImportError:
        logging.debug("PaddlePaddle not available")
        return False
    except Exception as e:
        logging.warning(f"Error initializing PaddlePaddle GPU: {e}")
        return False

def accelerate_image_processing(image: np.ndarray, 
                               denoise: bool = True, 
                               sharpen: bool = True, 
                               optimize_contrast: bool = False) -> np.ndarray:
    """
    Apply GPU acceleration to common image processing operations
    if GPU is available.
    
    Args:
        image: Numpy array containing the image
        denoise: Apply noise reduction
        sharpen: Apply sharpening
        optimize_contrast: Apply contrast enhancement
        
    Returns:
        np.ndarray: Processed image (may be unchanged if no acceleration applied)
    """
    if not is_gpu_available() or image is None:
        return image
        
    try:
        # Get OpenCV CUDA module
        cuda = get_gpu_enabled_opencv()
        if cuda is None:
            return image
            
        # Convert image to proper format if needed
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
            
        # Upload to GPU
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(image)
        
        # Apply processing based on parameters
        if denoise:
            # Apply bilateral filter for edge-preserving noise reduction
            # Parameters: d, sigmaColor, sigmaSpace
            gpu_img = cv2.cuda.bilateralFilter(gpu_img, 5, 75, 75)
            
        # Apply contrast enhancement if requested
        if optimize_contrast:
            try:
                # Need to handle color and grayscale images differently
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # For color images, convert to LAB and enhance L channel
                    gpu_lab = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2Lab)
                    
                    # Download for CLAHE (not available on GPU)
                    lab_img = gpu_lab.download()
                    
                    # Split into channels
                    l, a, b = cv2.split(lab_img)
                    
                    # Apply CLAHE to L channel
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    cl = clahe.apply(l)
                    
                    # Merge channels
                    enhanced = cv2.merge((cl, a, b))
                    
                    # Upload back to GPU
                    gpu_enhanced = cv2.cuda_GpuMat()
                    gpu_enhanced.upload(enhanced)
                    
                    # Convert back to BGR
                    gpu_img = cv2.cuda.cvtColor(gpu_enhanced, cv2.COLOR_Lab2BGR)
                else:
                    # For grayscale images, apply CLAHE directly
                    # Download to CPU for CLAHE (not available on GPU)
                    gray = gpu_img.download()
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(gray)
                    
                    # Upload back to GPU
                    gpu_img = cv2.cuda_GpuMat()
                    gpu_img.upload(enhanced)
            except Exception as e:
                logging.debug(f"GPU contrast enhancement failed: {e}")
        
        # Apply sharpening if requested
        if sharpen:
            try:
                # For sharpening we must work on the CPU
                # since custom kernels aren't supported on GPU
                cpu_img = gpu_img.download()
                
                # Create a sharpening kernel
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpened = cv2.filter2D(cpu_img, -1, kernel)
                
                # Upload sharpened image back to GPU
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(sharpened)
            except Exception as e:
                logging.debug(f"GPU sharpening failed: {e}")
        
        # Download result back to CPU
        result = gpu_img.download()
        return result
        
    except Exception as e:
        logging.warning(f"GPU image acceleration failed, using original image: {e}")
        return image

def batch_process_images(images: List[np.ndarray], 
                         process_fn=None, 
                         max_batch_size: int = 4,
                         max_workers: int = 2) -> List[np.ndarray]:
    """
    Process a batch of images efficiently with GPU acceleration.
    
    Args:
        images: List of image arrays to process
        process_fn: Function to apply to each image
                    If None, uses accelerate_image_processing
        max_batch_size: Maximum batch size for GPU processing
        max_workers: Maximum CPU workers for parallel processing
        
    Returns:
        List[np.ndarray]: List of processed images
    """
    if not images:
        return []
        
    # Use default processing function if none provided
    if process_fn is None:
        process_fn = accelerate_image_processing
        
    # If GPU isn't available, process in parallel on CPU
    if not is_gpu_available() or get_gpu_enabled_opencv() is None:
        import concurrent.futures
        
        # Process in parallel on CPU
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_fn, images))
        return results
        
    # For GPU processing, batch to avoid memory issues
    processed_images = []
    
    # Process in batches
    for i in range(0, len(images), max_batch_size):
        batch = images[i:i+max_batch_size]
        
        # Process each image in the batch
        batch_results = []
        for img in batch:
            try:
                processed = process_fn(img)
                batch_results.append(processed)
            except Exception as e:
                logging.warning(f"Error processing image in batch: {e}")
                # Return original on error
                batch_results.append(img)
                
        processed_images.extend(batch_results)
        
        # Explicit cleanup after each batch
        gc.collect()
        if is_gpu_available():
            try:
                cuda = get_gpu_enabled_opencv()
                if cuda is not None:
                    cuda.deviceReset()
            except:
                pass
                
    return processed_images

def enhance_text_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Apply GPU-accelerated image enhancement optimized for OCR.
    
    Args:
        image: Numpy array containing the image
        
    Returns:
        np.ndarray: Enhanced image
    """
    if not is_gpu_available() or image is None:
        # CPU fallback
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
                
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to RGB for OCR
            if len(image.shape) == 3:
                result = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                return result
            return thresh
        except Exception as e:
            logging.warning(f"CPU image enhancement for OCR failed: {e}")
            return image
    
    try:
        # Get OpenCV CUDA module
        cuda = get_gpu_enabled_opencv()
        if cuda is None:
            # Fall back to CPU
            return enhance_text_for_ocr(image)
            
        # Convert image to proper format and upload to GPU
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
            
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(image)
        
        # Convert to grayscale if color
        if len(image.shape) == 3:
            gray_gpu = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_gpu = gpu_img
            
        # Apply Gaussian blur to reduce noise
        blurred_gpu = cv2.cuda.GaussianBlur(gray_gpu, (5, 5), 0)
        
        # Download for adaptive threshold (not available on GPU)
        blurred = blurred_gpu.download()
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up noise
        kernel = np.ones((1, 1), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Upload back to GPU
        thresh_gpu = cv2.cuda_GpuMat()
        thresh_gpu.upload(thresh)
        
        # Convert back to RGB if input was RGB
        if len(image.shape) == 3:
            result_gpu = cv2.cuda.cvtColor(thresh_gpu, cv2.COLOR_GRAY2BGR)
            return result_gpu.download()
            
        return thresh_gpu.download()
        
    except Exception as e:
        logging.warning(f"GPU text enhancement for OCR failed: {e}")
        # Fall back to CPU processing
        return enhance_text_for_ocr(image)

def cleanup_gpu_resources():
    """
    Clean up GPU resources to prevent memory leaks.
    Call this before application shutdown.
    
    Returns:
        bool: True if cleanup was successful
    """
    if not is_gpu_available():
        return True
        
    try:
        # Reset OpenCV CUDA
        try:
            cuda = get_gpu_enabled_opencv()
            if cuda is not None:
                cuda.deviceReset()
                logging.debug("OpenCV CUDA resources released")
        except Exception as e:
            logging.debug(f"Error resetting OpenCV CUDA: {e}")
            
        # Clean up TensorFlow GPU resources if used
        try:
            import tensorflow as tf
            if tf.config.list_physical_devices('GPU'):
                tf.keras.backend.clear_session()
                logging.debug("TensorFlow GPU session cleared")
        except (ImportError, AttributeError):
            pass
            
        # Clean up PyTorch GPU cache if used
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.debug("PyTorch CUDA cache emptied")
        except (ImportError, AttributeError):
            pass
            
        # Clean up PaddlePaddle resources if used
        try:
            import paddle
            if paddle.device.is_compiled_with_cuda():
                # Clear cache and restore CPU mode
                paddle.device.cuda.empty_cache()
                logging.debug("PaddlePaddle CUDA cache emptied")
        except (ImportError, AttributeError):
            pass
            
        # Force full garbage collection
        gc.collect()
            
        return True
        
    except Exception as e:
        logging.warning(f"Error cleaning up GPU resources: {e}")
        return False

def force_memory_cleanup(wait_time: float = 0.5):
    """
    Force aggressive memory cleanup for GPU and system memory.
    
    Args:
        wait_time: Time to wait after cleanup to allow OS to reclaim memory
        
    Returns:
        None
    """
    # Clean up GPU resources first
    cleanup_gpu_resources()
    
    # Force garbage collection
    collected = gc.collect(2)  # Full collection with generation 2
    
    # Give the OS a moment to reclaim memory
    if wait_time > 0:
        time.sleep(wait_time)
        
    # Try one more time to ensure cleanup
    gc.collect()

def get_optimal_batch_size(image_height: int, image_width: int) -> int:
    """
    Determine optimal batch size for GPU processing based on 
    image dimensions and available GPU memory.
    
    Args:
        image_height: Height of the image in pixels
        image_width: Width of the image in pixels
        
    Returns:
        int: Recommended batch size
    """
    if not is_gpu_available():
        return 1  # Default to 1 for CPU processing
        
    # Get GPU info
    gpu_info = get_gpu_info()
    free_memory_mb = gpu_info.get('memory_free_mb', 0)
    
    # Estimate memory needed per image (in MB)
    # Assuming 3 channels, 8 bits per channel
    image_memory_mb = (image_height * image_width * 3) / (1024 * 1024)
    
    # Account for processing overhead (typically 2-3x the image size)
    processing_factor = 3.0
    
    # Calculate batch size, leaving 20% memory free for other operations
    safe_memory = free_memory_mb * 0.8
    batch_size = int(safe_memory / (image_memory_mb * processing_factor))
    
    # Ensure batch size is between 1 and a reasonable maximum
    batch_size = max(1, min(batch_size, 16))
    
    logging.debug(f"Calculated optimal batch size: {batch_size} for image size {image_width}x{image_height}")
    return batch_size

def estimate_gpu_memory_usage() -> Tuple[float, float, float]:
    """
    Estimate current GPU memory usage in MB.
    
    Returns:
        Tuple[float, float, float]: (used_mb, free_mb, total_mb)
    """
    if not is_gpu_available():
        return (0.0, 0.0, 0.0)
        
    try:
        # Try NVIDIA management library first
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            used_mb = info.used / (1024**2)
            free_mb = info.free / (1024**2)
            total_mb = info.total / (1024**2)
            
            pynvml.nvmlShutdown()
            return (used_mb, free_mb, total_mb)
        except ImportError:
            pass
            
        # Try PyTorch as fallback
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**2)
                reserved = torch.cuda.memory_reserved(0) / (1024**2)
                
                # Get total from nvidia-smi as PyTorch doesn't provide it directly
                total = 0.0
                gpu_info = get_gpu_info()
                total = gpu_info.get('memory_total_mb', 0)
                
                return (allocated, total - reserved, total)
        except ImportError:
            pass
            
        # If all else fails, try nvidia-smi directly
        gpu_info = get_gpu_info()
        total = gpu_info.get('memory_total_mb', 0)
        free = gpu_info.get('memory_free_mb', 0)
        used = total - free
        
        return (used, free, total)
    except Exception as e:
        logging.warning(f"Error estimating GPU memory usage: {e}")
        return (0.0, 0.0, 0.0)

class GPUResourceManager:
    """
    Unified GPU resource manager to centralize GPU-related tasks across the application.
    Provides methods for detection, memory management, cleanup, and reinitialization.
    
    Usage:
        # Get singleton instance
        gpu_manager = GPUResourceManager.get_instance()
        
        # Check if GPU is available
        if gpu_manager.is_available():
            # Initialize for specific framework
            gpu_manager.initialize_for_paddle()
            
        # Cleanup when done
        gpu_manager.cleanup()
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance of the GPU manager"""
        if cls._instance is None:
            cls._instance = GPUResourceManager()
        return cls._instance
    
    def __init__(self):
        """Initialize GPU resource manager and detect capabilities"""
        if GPUResourceManager._instance is not None:
            raise RuntimeError("GPUResourceManager is a singleton. Use get_instance() instead.")
            
        # Detect GPU status at initialization
        self._available = is_gpu_available()
        self._gpu_info = get_gpu_info() if self._available else {}
        self._frameworks = self._detect_frameworks()
        
        # Track initialization status for different frameworks
        self._initialized = {
            "paddle": False,
            "tensorflow": False,
            "pytorch": False,
            "opencv": False
        }
        
        # Memory management settings
        self._memory_fraction = 0.7  # Default memory fraction
        
        # Initialize OpenCV CUDA if available
        if self._available and "opencv" in self._frameworks:
            self._opencv_cuda = get_gpu_enabled_opencv()
            self._initialized["opencv"] = self._opencv_cuda is not None
        else:
            self._opencv_cuda = None
            
        logging.info(f"GPU Resource Manager initialized: GPU available: {self._available}")
        if self._available:
            logging.info(f"  Supported frameworks: {', '.join(self._frameworks)}")
            logging.info(f"  Memory: {self._gpu_info.get('memory_total_mb', 0)} MB total, "
                         f"{self._gpu_info.get('memory_free_mb', 0)} MB free")
    
    def _detect_frameworks(self):
        """Detect which GPU frameworks are available"""
        frameworks = []
        
        if self._available:
            try:
                import cv2
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    frameworks.append("opencv")
            except (ImportError, AttributeError):
                pass
                
            try:
                import paddle
                if paddle.device.is_compiled_with_cuda():
                    frameworks.append("paddle")
            except ImportError:
                pass
                
            try:
                import tensorflow as tf
                if tf.config.list_physical_devices('GPU'):
                    frameworks.append("tensorflow")
            except ImportError:
                pass
                
            try:
                import torch
                if torch.cuda.is_available():
                    frameworks.append("pytorch")
            except ImportError:
                pass
                
        return frameworks
    
    def is_available(self):
        """Check if GPU is available"""
        return self._available
    
    def get_info(self):
        """Get detailed GPU information"""
        # Refresh info
        if self._available:
            self._gpu_info = get_gpu_info()
        return self._gpu_info
    
    def get_memory_usage(self):
        """Get current GPU memory usage in MB"""
        if not self._available:
            return (0.0, 0.0, 0.0)
        return estimate_gpu_memory_usage()
    
    def set_memory_fraction(self, fraction=0.7):
        """Set memory fraction for all frameworks"""
        if not self._available or not (0.0 < fraction < 1.0):
            return False
            
        self._memory_fraction = fraction
        result = configure_gpu_memory(memory_fraction=fraction)
        return result
    
    def get_opencv_cuda(self):
        """Get OpenCV CUDA module if available"""
        if self._initialized["opencv"]:
            return self._opencv_cuda
        return None
    
    def initialize_for_paddle(self):
        """Initialize PaddleOCR with GPU acceleration"""
        if not self._available or "paddle" not in self._frameworks:
            return False
            
        if self._initialized["paddle"]:
            return True
            
        result = initialize_paddle_gpu()
        self._initialized["paddle"] = result
        return result
    
    def initialize_for_tensorflow(self):
        """Initialize TensorFlow with GPU acceleration"""
        if not self._available or "tensorflow" not in self._frameworks:
            return False
            
        if self._initialized["tensorflow"]:
            return True
            
        try:
            import tensorflow as tf
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                # Set memory growth to avoid allocating all memory at once
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                
                # Optionally limit memory usage
                if 0.0 < self._memory_fraction < 1.0:
                    for device in physical_devices:
                        tf.config.set_logical_device_configuration(
                            device,
                            [tf.config.LogicalDeviceConfiguration(
                                memory_limit=int(self._gpu_info.get('memory_total_mb', 0) * self._memory_fraction)
                            )]
                        )
                self._initialized["tensorflow"] = True
                logging.info(f"TensorFlow GPU initialized with memory fraction {self._memory_fraction}")
                return True
        except (ImportError, AttributeError, ValueError) as e:
            logging.debug(f"Could not initialize TensorFlow GPU: {e}")
            return False
    
    def initialize_for_pytorch(self):
        """Initialize PyTorch with GPU acceleration"""
        if not self._available or "pytorch" not in self._frameworks:
            return False
            
        if self._initialized["pytorch"]:
            return True
            
        try:
            import torch
            if torch.cuda.is_available():
                # Set to use a percentage of available memory
                if 0.0 < self._memory_fraction < 1.0:
                    torch.cuda.set_per_process_memory_fraction(self._memory_fraction)
                    logging.info(f"PyTorch GPU memory fraction set to {self._memory_fraction}")
                self._initialized["pytorch"] = True
                return True
        except (ImportError, AttributeError) as e:
            logging.debug(f"Could not initialize PyTorch GPU: {e}")
            return False
    
    def process_image(self, image, denoise=True, sharpen=True, optimize_contrast=False):
        """Process image with GPU acceleration if available"""
        if not self._available:
            return image
        return accelerate_image_processing(image, denoise, sharpen, optimize_contrast)
    
    def enhance_for_ocr(self, image):
        """Enhance image for OCR with GPU acceleration if available"""
        if not self._available:
            return image
        return enhance_text_for_ocr(image)
    
    def batch_process(self, images, process_fn=None, max_batch_size=4):
        """Process batch of images with GPU acceleration"""
        if not self._available:
            return images
            
        # Determine optimal batch size based on image size if applicable
        if len(images) > 0 and images[0] is not None:
            height, width = images[0].shape[:2]
            optimal_batch_size = get_optimal_batch_size(height, width)
            max_batch_size = min(max_batch_size, optimal_batch_size)
            
        return batch_process_images(images, process_fn, max_batch_size)
    
    def cleanup(self, force_gc=True):
        """Clean up GPU resources"""
        if not self._available:
            return True
            
        result = cleanup_gpu_resources()
        
        if force_gc:
            gc.collect()
        
        # Reset initialization state
        for framework in self._initialized:
            if framework != "opencv":  # Keep OpenCV status since it doesn't need re-initialization
                self._initialized[framework] = False
                
        return result
    
    def force_cleanup(self, wait_time=0.5):
        """Force aggressive cleanup of GPU resources"""
        if not self._available:
            return
        force_memory_cleanup(wait_time)
        
        # Reset initialization states
        for framework in self._initialized:
            if framework != "opencv":
                self._initialized[framework] = False