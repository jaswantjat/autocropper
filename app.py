"""
Complete Flask API for Card Rectification
Deploys the card-rectification algorithm as a REST API with comprehensive error handling
"""

import os
import sys
import cv2
import torch
import imutils
import base64
import tempfile
import numpy as np
import logging
import atexit
import threading
import zipfile
import img2pdf
import gc
from datetime import datetime
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from skimage import exposure, img_as_ubyte
from imutils.perspective import four_point_transform
from itertools import combinations
from torchvision import transforms

# Optional memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import the card rectification modules
from load_model import load_model
from models import get_model
from card_rectification import (
    rectify_card_image,
    validate_input_image,
    CardRectificationError,
    EdgeDetectionError,
    CornerDetectionError,
    TransformationError
)

# Global variables for model and configuration
trained_model = None
device = None
model_loaded = False
model_load_error = None
app_config = None

# Temporary file tracking for cleanup
temp_files = set()
temp_files_lock = threading.Lock()

# Initialize Flask app (configuration will be set later)
app = Flask(__name__)

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Set up comprehensive logging for debugging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.info("=== Logging initialized ===")
    return logger

def get_port():
    """Get port from environment with fallback."""
    return int(os.environ.get('PORT', 8080))

# Initialize logging immediately
logger = setup_logging()

# ============================================================================
# CONFIGURATION INITIALIZATION
# ============================================================================

def init_config():
    """Initialize configuration and set up Flask app."""
    global app_config, logger

    try:
        # Import and initialize configuration
        from config import get_config
        app_config = get_config()

        # Configure Flask app
        app.config['SECRET_KEY'] = app_config.SECRET_KEY
        app.config['MAX_CONTENT_LENGTH'] = app_config.MAX_CONTENT_LENGTH
        app.config['UPLOAD_FOLDER'] = app_config.get_upload_folder_str()

        # Configure logging
        logging.basicConfig(
            level=app_config.get_log_level(),
            format=app_config.LOG_FORMAT,
            filename=app_config.LOG_FILE if app_config.LOG_FILE else None,
            force=True  # Override any existing logging configuration
        )

        # Get logger after configuration
        logger = logging.getLogger(__name__)

        # Log configuration in debug mode
        if app_config.DEBUG:
            logger.info("Configuration initialized successfully")
            app_config.print_config()

        return True

    except Exception as e:
        print(f"Failed to initialize configuration: {e}")
        # Fall back to basic configuration
        app.config['SECRET_KEY'] = 'dev-secret-key'
        app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
        logging.basicConfig(level=logging.INFO, force=True)
        logger = logging.getLogger(__name__)
        logger.error(f"Using fallback configuration due to error: {e}")
        return False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    if app_config:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app_config.ALLOWED_EXTENSIONS
    else:
        # Fallback extensions
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'bmp'}

def add_temp_file(filepath):
    """Add a temporary file to the cleanup list."""
    with temp_files_lock:
        temp_files.add(filepath)

def remove_temp_file(filepath):
    """Remove a temporary file and clean it up."""
    with temp_files_lock:
        if filepath in temp_files:
            temp_files.remove(filepath)

    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.debug(f"Cleaned up temporary file: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to remove temporary file {filepath}: {e}")

def get_temp_file_path(suffix='.tmp'):
    """Get a temporary file path in the configured temp directory."""
    import tempfile
    if app_config:
        return tempfile.mktemp(suffix=suffix, dir=app_config.get_temp_folder_str())
    else:
        # Fallback to system temp directory
        return tempfile.mktemp(suffix=suffix)

def cleanup_all_temp_files():
    """Clean up all tracked temporary files."""
    with temp_files_lock:
        files_to_remove = list(temp_files)

    for filepath in files_to_remove:
        remove_temp_file(filepath)

    logger.info(f"Cleaned up {len(files_to_remove)} temporary files")

def check_memory():
    """Check available memory and return True if sufficient."""
    if not PSUTIL_AVAILABLE:
        return True

    try:
        memory = psutil.virtual_memory()
        # Allow configurable memory threshold for testing
        threshold = int(os.environ.get('MEMORY_THRESHOLD', '85'))
        if memory.percent > threshold:
            logger.warning(f"High memory usage: {memory.percent}% (threshold: {threshold}%)")
            gc.collect()  # Force garbage collection
            return False
        return True
    except Exception as e:
        logger.warning(f"Memory check failed: {e}")
        return True

def load_model_with_logging():
    """Load model with comprehensive logging."""
    global trained_model, device, model_loaded, model_load_error

    logger.info("=== Starting model loading ===")

    try:
        # Get model path from configuration
        if app_config:
            model_path = app_config.get_model_path_str()
            force_cpu = app_config.FORCE_CPU
        else:
            model_path = "CRDN1000.pkl"
            force_cpu = os.environ.get('FORCE_CPU', 'false').lower() == 'true'

        logger.info(f"Checking for model file: {model_path}")

        # Check if model file exists, try to download if missing
        if not os.path.exists(model_path):
            logger.warning(f"Model file {model_path} not found. Attempting to download...")
            try:
                from download_model import download_model
                if download_model():
                    logger.info("Model downloaded successfully")
                else:
                    raise FileNotFoundError("Failed to download model file")
            except Exception as download_error:
                logger.error(f"Failed to download model: {download_error}")
                raise FileNotFoundError(f"Model file {model_path} not found and download failed")
        else:
            logger.info("Model file found, attempting to load...")

        # Import and load the model
        logger.info("Importing load_model module...")
        from load_model import load_model

        logger.info("Loading model...")
        trained_model, device = load_model()

        # Force CPU if configured
        if force_cpu and device.type == 'cuda':
            logger.info("Forcing CPU usage as configured")
            device = torch.device('cpu')
            trained_model = trained_model.to(device)

        model_loaded = True
        model_load_error = None

        logger.info(f"✓ Model loaded successfully on device: {device}")
        logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available() and not force_cpu:
            logger.info(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("✓ Running on CPU")

    except Exception as e:
        model_loaded = False
        model_load_error = str(e)
        logger.error(f"✗ Failed to load model: {e}")
        logger.error(f"✗ Model load error details: {type(e).__name__}: {str(e)}")
        # Don't exit - let the app start but return errors for processing requests

    logger.info("=== Model loading complete ===")

def init_model():
    """Initialize the card rectification model with proper error handling."""
    load_model_with_logging()

def validate_image(image_data):
    """Validate that the image data is valid and can be processed."""
    if image_data is None:
        raise ValueError("Invalid image data")

    if len(image_data.shape) != 3:
        raise ValueError("Image must be a 3-channel color image")

    height, width = image_data.shape[:2]
    if height < 100 or width < 100:
        raise ValueError("Image too small (minimum 100x100 pixels)")

    if height > 5000 or width > 5000:
        raise ValueError("Image too large (maximum 5000x5000 pixels)")

    return True

# ============================================================================
# PDF CONVERSION UTILITIES
# ============================================================================

def images_to_pdf(image_list, page_size=None, quality=95):
    """
    Convert a list of images to a single PDF.

    Args:
        image_list: List of image data (numpy arrays or PIL Images)
        page_size: Page size for PDF (e.g., 'A4', 'letter', or (width, height))
        quality: JPEG quality for compression (1-100)

    Returns:
        PDF bytes
    """
    try:
        logger.debug(f"Converting {len(image_list)} images to PDF")

        # Convert images to bytes
        image_bytes_list = []
        for i, img in enumerate(image_list):
            if isinstance(img, np.ndarray):
                # Convert numpy array to PIL Image
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif isinstance(img, Image.Image):
                pil_img = img
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

            # Convert to bytes
            img_buffer = BytesIO()
            pil_img.save(img_buffer, format='JPEG', quality=quality, optimize=True)
            image_bytes_list.append(img_buffer.getvalue())
            logger.debug(f"Converted image {i+1} to JPEG ({len(img_buffer.getvalue())} bytes)")

        # Configure page size
        layout_kwargs = {}
        if page_size:
            if isinstance(page_size, str):
                if page_size.lower() == 'a4':
                    layout_kwargs['pagesize'] = (img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297))
                elif page_size.lower() == 'letter':
                    layout_kwargs['pagesize'] = (img2pdf.in_to_pt(8.5), img2pdf.in_to_pt(11))
                elif page_size.lower() == 'legal':
                    layout_kwargs['pagesize'] = (img2pdf.in_to_pt(8.5), img2pdf.in_to_pt(14))
            elif isinstance(page_size, (tuple, list)) and len(page_size) == 2:
                layout_kwargs['pagesize'] = page_size

        # Create PDF
        if layout_kwargs:
            pdf_bytes = img2pdf.convert(image_bytes_list, **layout_kwargs)
        else:
            pdf_bytes = img2pdf.convert(image_bytes_list)

        logger.info(f"Successfully created PDF with {len(image_list)} images ({len(pdf_bytes)} bytes)")
        return pdf_bytes

    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        raise Exception(f"PDF conversion failed: {e}")

def create_zip_with_images_and_pdf(processed_images, original_filenames, page_size='A4', quality=90):
    """
    Create a ZIP file containing individual processed images and a combined PDF.

    Args:
        processed_images: List of processed image arrays
        original_filenames: List of original filenames

    Returns:
        ZIP file bytes
    """
    try:
        logger.debug(f"Creating ZIP with {len(processed_images)} images and PDF")

        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add individual processed images
            for i, (img, original_name) in enumerate(zip(processed_images, original_filenames)):
                # Generate filename for processed image
                base_name = os.path.splitext(secure_filename(original_name))[0]
                processed_filename = f"processed_{base_name}.png"

                # Convert image to PNG bytes
                success, buffer = cv2.imencode('.png', img)
                if not success:
                    raise Exception(f"Failed to encode image {i+1}")

                # Add to ZIP
                zip_file.writestr(processed_filename, buffer.tobytes())
                logger.debug(f"Added {processed_filename} to ZIP")

            # Create and add PDF
            pdf_bytes = images_to_pdf(processed_images, page_size=page_size, quality=quality)
            zip_file.writestr("processed_cards.pdf", pdf_bytes)
            logger.debug("Added PDF to ZIP")

        zip_buffer.seek(0)
        result_bytes = zip_buffer.getvalue()

        logger.info(f"Successfully created ZIP file ({len(result_bytes)} bytes)")
        return result_bytes

    except Exception as e:
        logger.error(f"ZIP creation failed: {e}")
        raise Exception(f"ZIP creation failed: {e}")

def parse_pdf_options(request_data):
    """Parse PDF options from request data."""
    options = {
        'page_size': request_data.get('page_size', 'A4'),
        'quality': int(request_data.get('quality', 90))
    }

    # Validate quality
    if not 1 <= options['quality'] <= 100:
        options['quality'] = 90

    # Validate page size
    valid_page_sizes = ['A4', 'letter', 'legal']
    if options['page_size'] not in valid_page_sizes:
        options['page_size'] = 'A4'

    return options

# ============================================================================
# CARD RECTIFICATION INTEGRATION
# ============================================================================
# Note: Core rectification algorithm is now in card_rectification.py module

# All card rectification algorithm functions have been moved to card_rectification.py module
# This provides better separation of concerns and easier maintenance

def rectify_card(image):
    """
    Main card rectification function using the integrated algorithm.

    Args:
        image: Input image as numpy array (BGR format)

    Returns:
        Rectified card image as numpy array

    Raises:
        Exception: If card rectification fails
    """
    global trained_model, device, model_loaded

    try:
        # Check if model is loaded
        if not model_loaded:
            logger.warning("Model not loaded, using traditional edge detection only")
            model_to_use = None
            device_to_use = None
        else:
            model_to_use = trained_model
            device_to_use = device

        # Validate input image using the new module
        validate_input_image(image)

        # Use the integrated card rectification algorithm
        result = rectify_card_image(image, model_to_use, device_to_use)

        logger.info("✓ Card rectification completed successfully using integrated algorithm")
        return result

    except CardRectificationError as e:
        logger.error(f"Card rectification failed: {e}")
        raise Exception(str(e))
    except Exception as e:
        logger.error(f"Unexpected error in card rectification: {e}")
        raise Exception(f"Card rectification failed: {e}")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def api_info():
    """
    Root endpoint providing API information and available endpoints.
    """
    return jsonify({
        'name': 'Card Rectification API',
        'version': '2.0.0',
        'description': 'REST API for ID card rectification using deep learning',
        'endpoints': {
            'health': {
                'method': 'GET',
                'path': '/health',
                'description': 'Check API and model status'
            },
            'process_file': {
                'method': 'POST',
                'path': '/api/process-id',
                'description': 'Process ID card from uploaded file',
                'content_type': 'multipart/form-data',
                'parameters': {
                    'file': 'Image file (PNG, JPG, JPEG, BMP)'
                }
            },
            'process_base64': {
                'method': 'POST',
                'path': '/api/process-id-base64',
                'description': 'Process ID card from base64 encoded image',
                'content_type': 'application/json',
                'parameters': {
                    'image': 'Base64 encoded image string'
                }
            },
            'process_and_pdf': {
                'method': 'POST',
                'path': '/api/process-and-pdf',
                'description': 'Process one or more ID cards and return as PDF',
                'content_type': 'multipart/form-data',
                'parameters': {
                    'files': 'One or more image files',
                    'page_size': 'Optional: A4, letter, legal (default: A4)',
                    'quality': 'Optional: JPEG quality 1-100 (default: 90)'
                }
            },
            'process_multiple': {
                'method': 'POST',
                'path': '/api/process-multiple',
                'description': 'Process multiple ID cards and return ZIP with images and PDF',
                'content_type': 'multipart/form-data',
                'parameters': {
                    'files': 'Multiple image files',
                    'page_size': 'Optional: A4, letter, legal (default: A4)',
                    'quality': 'Optional: JPEG quality 1-100 (default: 90)'
                }
            }
        },
        'supported_formats': list(app_config.ALLOWED_EXTENSIONS) if app_config else ['png', 'jpg', 'jpeg', 'bmp'],
        'max_file_size': f"{(app_config.MAX_CONTENT_LENGTH // (1024*1024))}MB" if app_config else '16MB',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify API status and model readiness.

    Returns:
        JSON response with API status, model status, and system information
    """
    try:
        model_status = "ready" if model_loaded else "error"
        model_info = {
            "loaded": model_loaded,
            "device": str(device) if device else "unknown",
            "cuda_available": torch.cuda.is_available(),
            "error": model_load_error if model_load_error else None
        }

        if torch.cuda.is_available():
            model_info["gpu_name"] = torch.cuda.get_device_name(0)
            model_info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"

        # Add memory info if available
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                model_info["system_memory_usage"] = f"{memory.percent}%"
                model_info["system_memory_available"] = f"{memory.available / (1024**3):.1f}GB"
            except Exception:
                pass

        return jsonify({
            'status': 'healthy',
            'message': 'Card Rectification API is running',
            'version': '2.0.0',
            'model_status': model_status,
            'model_info': model_info,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Health check failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/live', methods=['GET'])
def liveness_check():
    """
    Lightweight liveness probe. Does not depend on model readiness.
    Suitable for container/platform liveness checks.
    """
    return jsonify({
        'status': 'ok',
        'message': 'Service is alive',
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/ready', methods=['GET'])
def readiness_check():
    """
    Readiness probe. Returns 200 only when the model is loaded and API is ready to serve.
    Returns 503 otherwise so platforms like Railway can avoid routing traffic.
    """
    if model_loaded:
        return jsonify({
            'status': 'ready',
            'message': 'Model is loaded and API is ready',
            'version': '2.0.0',
            'timestamp': datetime.now().isoformat()
        }), 200
    else:
        return jsonify({
            'status': 'not_ready',
            'message': 'Model is not loaded yet',
            'error': model_load_error,
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/api/process-id', methods=['POST'])
def process_id_upload():
    """
    Process ID card from uploaded file.

    Accepts multipart/form-data with 'file' field containing image.
    Returns rectified card image as PNG file.
    """
    temp_file_path = None

    try:
        # Check memory before processing
        if not check_memory():
            return jsonify({
                'error': 'Insufficient memory available',
                'details': 'System memory usage too high'
            }), 503

        # Check if model is loaded
        if not model_loaded:
            return jsonify({
                'error': 'Model not available',
                'details': model_load_error
            }), 503

        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided in request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'allowed_types': list(app_config.ALLOWED_EXTENSIONS) if app_config else ['png', 'jpg']
            }), 400

        # Create temporary file for processing
        temp_file_path = tempfile.mktemp(suffix='.tmp')
        add_temp_file(temp_file_path)

        # Read and validate image
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({'error': 'Empty file'}), 400

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid or corrupted image file'}), 400

        logger.info(f"Processing uploaded file: {file.filename}, size: {image.shape}")

        # Rectify the card
        result = rectify_card(image)

        # Convert result to bytes
        success, buffer = cv2.imencode('.png', result)
        if not success:
            raise Exception("Failed to encode result image")

        result_bytes = buffer.tobytes()

        logger.info(f"✓ Successfully processed {file.filename}")

        # Force cleanup after processing
        gc.collect()

        # Return as file download
        return send_file(
            BytesIO(result_bytes),
            mimetype='image/png',
            as_attachment=True,
            download_name=f'rectified_{secure_filename(file.filename)}.png'
        )

    except Exception as e:
        logger.error(f"File upload processing failed: {e}")
        return jsonify({
            'error': 'Processing failed',
            'details': str(e)
        }), 500

    finally:
        # Clean up temporary file
        if temp_file_path:
            remove_temp_file(temp_file_path)

@app.route('/api/process-id-base64', methods=['POST'])
def process_id_base64():
    """
    Process ID card from base64 encoded image.

    Accepts JSON with 'image' field containing base64 encoded image.
    Returns JSON with rectified card image as base64 string.
    """
    try:
        # Check if model is loaded
        if not model_loaded:
            return jsonify({
                'success': False,
                'error': 'Model not available',
                'details': model_load_error
            }), 503

        # Validate request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided in JSON body'
            }), 400

        # Decode base64 image
        try:
            image_data = data['image']

            # Handle data URL format (data:image/jpeg;base64,...)
            if image_data.startswith('data:image'):
                if ',' not in image_data:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid data URL format'
                    }), 400
                image_data = image_data.split(',')[1]

            # Decode base64
            image_bytes = base64.b64decode(image_data)
            if len(image_bytes) == 0:
                return jsonify({
                    'success': False,
                    'error': 'Empty image data'
                }), 400

            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return jsonify({
                    'success': False,
                    'error': 'Invalid or corrupted image data'
                }), 400

        except Exception as e:
            logger.warning(f"Base64 decode failed: {e}")
            return jsonify({
                'success': False,
                'error': 'Invalid base64 image data',
                'details': str(e)
            }), 400

        logger.info(f"Processing base64 image, size: {image.shape}")

        # Rectify the card
        result = rectify_card(image)

        # Convert result to base64
        success, buffer = cv2.imencode('.png', result)
        if not success:
            raise Exception("Failed to encode result image")

        result_base64 = base64.b64encode(buffer).decode('utf-8')

        logger.info("✓ Successfully processed base64 image")

        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{result_base64}',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Base64 processing failed: {e}")
        return jsonify({
            'success': False,
            'error': 'Processing failed',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/process-and-pdf', methods=['POST'])
def process_and_pdf():
    """
    Process one or more ID cards and return as a single PDF.

    Accepts multipart/form-data with multiple 'files' fields and optional parameters.
    Returns a PDF file containing all processed cards.
    """
    temp_file_paths = []

    try:
        # Check if model is loaded
        if not model_loaded:
            return jsonify({
                'error': 'Model not available',
                'details': model_load_error
            }), 503

        # Get files from request
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files provided in request'}), 400

        # Parse PDF options
        pdf_options = parse_pdf_options(request.form)

        logger.info(f"Processing {len(files)} files for PDF conversion")
        logger.info(f"PDF options: {pdf_options}")

        processed_images = []

        # Process each file
        for i, file in enumerate(files):
            if file.filename == '':
                continue

            if not allowed_file(file.filename):
                return jsonify({
                    'error': f'Invalid file type for file {i+1}: {file.filename}',
                    'allowed_types': list(app_config.ALLOWED_EXTENSIONS) if app_config else ['png', 'jpg']
                }), 400

            # Read and validate image
            image_bytes = file.read()
            if len(image_bytes) == 0:
                return jsonify({'error': f'Empty file: {file.filename}'}), 400

            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return jsonify({'error': f'Invalid or corrupted image file: {file.filename}'}), 400

            logger.info(f"Processing file {i+1}/{len(files)}: {file.filename}, size: {image.shape}")

            # Rectify the card
            result = rectify_card(image)
            processed_images.append(result)

        if not processed_images:
            return jsonify({'error': 'No valid images to process'}), 400

        # Convert to PDF
        pdf_bytes = images_to_pdf(
            processed_images,
            page_size=pdf_options['page_size'],
            quality=pdf_options['quality']
        )

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'processed_cards_{timestamp}.pdf'

        logger.info(f"✓ Successfully created PDF with {len(processed_images)} processed cards")

        # Return PDF file
        return send_file(
            BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        return jsonify({
            'error': 'PDF processing failed',
            'details': str(e)
        }), 500

    finally:
        # Clean up temporary files
        for temp_file_path in temp_file_paths:
            remove_temp_file(temp_file_path)

@app.route('/api/process-multiple', methods=['POST'])
def process_multiple():
    """
    Process multiple ID cards and return ZIP containing individual images and PDF.

    Accepts multipart/form-data with multiple 'files' fields and optional parameters.
    Returns a ZIP file containing individual processed images and a combined PDF.
    """
    temp_file_paths = []

    try:
        # Check if model is loaded
        if not model_loaded:
            return jsonify({
                'error': 'Model not available',
                'details': model_load_error
            }), 503

        # Get files from request
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files provided in request'}), 400

        # Parse PDF options
        pdf_options = parse_pdf_options(request.form)

        logger.info(f"Processing {len(files)} files for ZIP with images and PDF")
        logger.info(f"PDF options: {pdf_options}")

        processed_images = []
        original_filenames = []

        # Process each file
        for i, file in enumerate(files):
            if file.filename == '':
                continue

            if not allowed_file(file.filename):
                return jsonify({
                    'error': f'Invalid file type for file {i+1}: {file.filename}',
                    'allowed_types': list(app_config.ALLOWED_EXTENSIONS) if app_config else ['png', 'jpg']
                }), 400

            # Read and validate image
            image_bytes = file.read()
            if len(image_bytes) == 0:
                return jsonify({'error': f'Empty file: {file.filename}'}), 400

            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return jsonify({'error': f'Invalid or corrupted image file: {file.filename}'}), 400

            logger.info(f"Processing file {i+1}/{len(files)}: {file.filename}, size: {image.shape}")

            # Rectify the card
            result = rectify_card(image)
            processed_images.append(result)
            original_filenames.append(file.filename)

        if not processed_images:
            return jsonify({'error': 'No valid images to process'}), 400

        # Create ZIP with images and PDF
        zip_bytes = create_zip_with_images_and_pdf(
            processed_images,
            original_filenames,
            page_size=pdf_options['page_size'],
            quality=pdf_options['quality']
        )

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'processed_cards_{timestamp}.zip'

        logger.info(f"✓ Successfully created ZIP with {len(processed_images)} processed cards")

        # Return ZIP file
        return send_file(
            BytesIO(zip_bytes),
            mimetype='application/zip',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.error(f"Multiple file processing failed: {e}")
        return jsonify({
            'error': 'Multiple file processing failed',
            'details': str(e)
        }), 500

    finally:
        # Clean up temporary files
        for temp_file_path in temp_file_paths:
            remove_temp_file(temp_file_path)

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def payload_too_large(e):
    """Handle file size limit exceeded."""
    return jsonify({
        'error': 'File too large',
        'details': 'Maximum file size is 16MB',
        'max_size': '16MB'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle endpoint not found."""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /health',
            'POST /api/process-id',
            'POST /api/process-id-base64',
            'POST /api/process-and-pdf',
            'POST /api/process-multiple'
        ]
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    """Handle method not allowed."""
    return jsonify({
        'error': 'Method not allowed',
        'details': 'Check the HTTP method and endpoint'
    }), 405

@app.errorhandler(500)
def internal_server_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {e}")
    return jsonify({
        'error': 'Internal server error',
        'details': 'An unexpected error occurred'
    }), 500

# ============================================================================
# APPLICATION STARTUP
# ============================================================================

def register_cleanup():
    """Register cleanup functions to run on application exit."""
    atexit.register(cleanup_all_temp_files)
    logger.info("✓ Cleanup handlers registered")

# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

def initialize_app():
    """Initialize the application with comprehensive logging."""
    logger.info("=== Initializing Card Rectification API ===")

    # Log environment
    logger.info(f"Environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'unknown')}")
    logger.info(f"Port: {get_port()}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python version: {sys.version}")

    # List files in current directory
    try:
        files = os.listdir('.')
        logger.info(f"Files in directory: {files}")
    except Exception as e:
        logger.error(f"Failed to list directory: {e}")

    # Initialize configuration
    logger.info("Initializing configuration...")
    config_success = init_config()
    if not config_success:
        logger.warning("Using fallback configuration")

    # Register cleanup handlers
    register_cleanup()

    # Load model
    logger.info("Loading model...")
    init_model()

    if model_loaded:
        logger.info("✓ Model loaded successfully")
    else:
        logger.warning(f"⚠️ Model loading failed: {model_load_error}")

    logger.info("=== Application initialization complete ===")

def main():
    """Main application entry point."""
    try:
        # Initialize the application
        initialize_app()

        # Start the Flask application using configuration
        if app_config:
            host = app_config.HOST
            port = app_config.PORT
            debug_mode = app_config.DEBUG
        else:
            # Fallback values
            host = os.environ.get('HOST', '0.0.0.0')
            port = get_port()
            debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'

        logger.info(f"🚀 Starting Flask app on {host}:{port} (debug={debug_mode})")
        app.run(
            host=host,
            port=port,
            debug=debug_mode,
            threaded=True
        )

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        sys.exit(1)
    finally:
        cleanup_all_temp_files()

# Initialize the app when this module is imported (for gunicorn)
try:
    initialize_app()
except Exception as e:
    logger.error(f"Failed to initialize app during import: {e}")

if __name__ == '__main__':
    main()
