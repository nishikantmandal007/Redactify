#!/usr/bin/env python3
# Redactify/web/routes.py

import os
import re
import logging
from flask import (
    Blueprint, render_template, redirect, url_for, flash,
    send_from_directory, request, jsonify, current_app
)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge, NotFound
from celery.result import AsyncResult

# Import from our new modular structure
from ..core.config import UPLOAD_DIR, TEMP_DIR, MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB, TEMP_FILE_MAX_AGE_SECONDS
from ..services.celery_service import celery
from ..services.tasks import perform_redaction
from ..services.cleanup import cleanup_temp_files
from ..core.analyzers import get_pii_types as get_pii_choices_from_util
from ..processors.qr_code_processor import get_supported_barcode_types
from .forms import UploadForm

# Create a blueprint for the web routes
bp = Blueprint('main', __name__)

# --- Helper Function ---
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    has_dot = '.' in filename
    if not has_dot:
        logging.debug(f"allowed_file check: No '.' found in filename '{filename}'")
        return False
        
    allowed = current_app.config['UPLOAD_EXTENSIONS']  # Use current_app context
    ext = os.path.splitext(filename)[1].lower()
    is_allowed = ext in allowed
    logging.debug(f"allowed_file check: filename='{filename}', extension='{ext}', allowed={allowed}, result={is_allowed}")
    return is_allowed
# --- End Helper Function ---


# --- Flask Routes ---

@bp.route('/', methods=['GET'])
def index():
    """Render the main upload page."""
    form = UploadForm()
    # Dynamically load choices for PII types
    try:
        pii_choices_list = get_pii_choices_from_util()  # Gets defaults + custom + QR_CODE
        # Format choices for the WTForms SelectMultipleField
        form.pii_types.choices = [(ptype, ptype.replace('_', ' ').title()) for ptype in pii_choices_list]
        logging.debug(f"Populated PII choices in index route: {form.pii_types.choices}")
        
        # Dynamically load barcode types
        barcode_types_dict = get_supported_barcode_types()
        form.barcode_types.choices = [(code, desc) for code, desc in barcode_types_dict.items()]
        logging.debug(f"Populated barcode type choices: {form.barcode_types.choices}")
    except Exception as e:
        logging.error(f"Failed to load choices: {e}", exc_info=True)
        flash("Error loading options.", "error")
        form.pii_types.choices = []  # Set empty list on error
        form.barcode_types.choices = []  # Set empty list on error

    return render_template('index.html', form=form, max_size_mb=MAX_FILE_SIZE_MB, TEMP_FILE_MAX_AGE_SECONDS=TEMP_FILE_MAX_AGE_SECONDS)


@bp.route('/process_pdf', methods=['POST'])
def process_ajax():
    """Handles AJAX form submission, saves file, queues Celery task."""
    # File handling first
    if 'pdf_file' not in request.files:
        logging.warning("AJAX POST /process_pdf: No file part in request.files.")
        return jsonify({'error': 'No file part in the request.'}), 400

    pdf_file = request.files['pdf_file']

    if pdf_file.filename == '':
        logging.warning("AJAX POST /process_pdf: No filename provided.")
        return jsonify({'error': 'No file selected.'}), 400

    # Input validation
    if not pdf_file or not pdf_file.filename.lower().endswith('.pdf'):
        logging.warning(f"AJAX POST /process_pdf: Invalid file type or empty file (Direct Check): {pdf_file.filename}")
        return jsonify({'error': 'Invalid file type. Please upload a PDF.'}), 400

    # Secure filename and path
    filename = secure_filename(pdf_file.filename)
    if not filename:
        logging.warning("AJAX POST /process_pdf: Invalid secure filename generated.")
        return jsonify({'error': 'Invalid filename.'}), 400

    pdf_path = os.path.join(UPLOAD_DIR, filename)  # Use UPLOAD_DIR from config

    # Avoid overwriting - generate unique name if exists
    counter = 1
    base, ext = os.path.splitext(pdf_path)
    while os.path.exists(pdf_path):
        pdf_path = f"{base}_{counter}{ext}"
        filename = os.path.basename(pdf_path)  # Update filename if changed
        counter += 1
        if counter > 100:  # Safety break
            logging.error("AJAX POST /process_pdf: Could not generate unique filename.")
            return jsonify({'error': 'Could not save file, name conflict.'}), 500

    # Try saving the file
    try:
        pdf_file.save(pdf_path)
        logging.info(f"AJAX POST /process_pdf: File saved to {pdf_path}")
    except RequestEntityTooLarge:
        # Caught by error handler, but log here too
        logging.warning(f"AJAX POST /process_pdf: File too large: {filename}")
        # The error handler will return the response
        raise  # Re-raise to trigger the handler
    except Exception as e:
        logging.error(f"AJAX POST /process_pdf: Failed to save uploaded file {filename}: {e}", exc_info=True)
        return jsonify({'error': f'Error saving file: {e}'}), 500

    # Get other form data
    pii_types_selected = request.form.getlist('pii_types')
    keyword_data = request.form.get('keyword_rules', '')
    regex_data = request.form.get('regex_rules', '')

    # Validate PII types against known list
    valid_pii_types = get_pii_choices_from_util()
    pii_types_selected = [ptype for ptype in pii_types_selected if ptype in valid_pii_types]

    custom_rules = {}
    if keyword_data:
        custom_rules["keyword"] = [kw.strip() for kw in keyword_data.splitlines() if kw.strip()]
    if regex_data:
        custom_rules["regex"] = []
        for rx in regex_data.splitlines():
            rx_strip = rx.strip()
            if rx_strip:
                try:
                    re.compile(rx_strip)  # Try compiling regex
                    custom_rules["regex"].append(rx_strip)
                except re.error as regex_err:
                    logging.warning(f"Invalid regex pattern submitted and ignored: {rx_strip} - Error: {regex_err}")
                    pass  # Option to ignore

    # Queue task
    try:
        task = perform_redaction.delay(pdf_path, pii_types_selected, custom_rules)
        logging.info(f"AJAX POST /process_pdf: Task {task.id} queued for file {filename}")
        return jsonify({'task_id': task.id}), 202
    except Exception as e:
        logging.error(f"AJAX POST /process_pdf: Failed to queue task for {filename}: {e}", exc_info=True)
        # Clean up saved file if queuing fails
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except OSError as del_err:
                logging.warning(f"Could not delete file {pdf_path} after task queue failure: {del_err}")
        return jsonify({'error': f'Error queueing redaction task. Is the background service running?'}), 500


@bp.route('/task_status/<task_id>')
def task_status(task_id):
    """Provides task status updates for AJAX polling."""
    try:
        # Ensure celery app context is used for result backend config
        task = AsyncResult(task_id, app=celery)
    except Exception as e:
        logging.error(f"Error creating AsyncResult for task_id {task_id}: {e}")
        return jsonify({'state': 'ERROR', 'status': 'Invalid Task ID format or backend error.'}), 404

    response = {'state': task.state, 'status': 'Waiting...', 'progress': 0, 'result': None}

    if not task:  # Should not happen if AsyncResult created without error
        response['state'] = 'ERROR'
        response['status'] = 'Task ID not found or invalid.'
        return jsonify(response), 404

    # Update response based on task state
    # Get task info safely, default to empty dict if None
    info = task.info if task.info is not None else {}

    if task.state == 'PENDING':
        response['status'] = 'Task is waiting in queue...'
    elif task.state == 'STARTED':
        # Ensure info is a dict before using .get
        response['status'] = info.get('status', 'Task has started processing...') if isinstance(info, dict) else 'Task has started...'
    elif task.state == 'PROGRESS':
        if isinstance(info, dict):
            response['progress'] = int((info.get('current', 0) / info.get('total', 1)) * 100) if info.get('total', 1) > 0 else 0
            response['status'] = info.get('status', 'Processing...')
        else:
            response['status'] = 'Processing... (progress info unavailable)'  # Fallback
            response['progress'] = 50  # Indicate some progress
    elif task.state == 'SUCCESS':
        response['progress'] = 100
        response['status'] = info.get('status', 'Task completed successfully!') if isinstance(info, dict) else 'Task completed!'
        response['result'] = info.get('result') if isinstance(info, dict) else None  # Include result filename
    elif task.state == 'FAILURE':
        response['progress'] = 100  # Indicate finished but failed
        exc_type_name = 'Unknown Error'
        # Check if info is an Exception object
        if isinstance(info, Exception):
            exc_type_name = type(info).__name__
            # Log the actual exception on the server side
            logging.error(f"Reporting FAILURE for task {task_id}: Exception Type={exc_type_name}, Exception Args={info.args}")
        elif isinstance(info, dict):  # Check if it's the dict we set in the task's except block
            exc_type_name = info.get('exc_type', 'Error')
            logging.error(f"Reporting FAILURE for task {task_id}: Type={exc_type_name}, Msg={info.get('exc_message', 'N/A')}")
        else:
            # Info is something else unexpected
            logging.error(f"Reporting FAILURE for task {task_id}: Unexpected info type: {type(info).__name__}, value: {str(info)[:200]}")

        # User-facing message is kept simple
        response['status'] = f"Task failed: {exc_type_name}. Check server logs for details."
    elif task.state == 'RETRY':
        retries = info.get('retries', 0) if isinstance(info, dict) else 0
        exc_type = info.get('exc_type', 'Error') if isinstance(info, dict) else 'Unknown'
        try:
            current_retry = retries + 1
            max_retries = task.max_retries if hasattr(task, 'max_retries') and task.max_retries is not None else 'N/A'
            status_suffix = f" (Retry {current_retry}/{max_retries})"
        except Exception:
            status_suffix = " (Retrying...)"
        response['status'] = f"Task issue{status_suffix}. Reason: {exc_type}"
        response['progress'] = info.get('progress_before_retry', 0) if isinstance(info, dict) else 0
    else:  # Handle other states (REVOKED, etc.)
        response['status'] = f"Task state: {task.state}"

    return jsonify(response)


@bp.route('/result/<task_id>')
def result(task_id):
    """Serves the final redacted PDF file based on task ID."""
    try:
        task = AsyncResult(task_id, app=celery)  # Pass celery app instance
    except Exception as e:
        logging.error(f"Error creating AsyncResult for task_id {task_id} in /result: {e}")
        flash("Invalid Task ID format.", "error")
        return redirect(url_for('main.index'))

    if not task:
        flash(f"Unknown task ID: {task_id}", "error")
        return redirect(url_for('main.index'))

    if task.state == 'SUCCESS':
        result_info = task.info or {}
        redacted_filename = result_info.get('result')
        if redacted_filename:
            safe_filename = secure_filename(redacted_filename)
            if not safe_filename:
                logging.error(f"Invalid result filename format for task {task_id}: {redacted_filename}")
                flash("Result filename is invalid.", "error")
                return redirect(url_for('main.index'))
            try:
                logging.info(f"Attempting to serve result file: {safe_filename} from {TEMP_DIR}")
                full_path = os.path.join(TEMP_DIR, safe_filename)
                # Basic path traversal check
                if not os.path.abspath(full_path).startswith(os.path.abspath(TEMP_DIR)):
                    logging.error(f"Path traversal attempt detected for task {task_id}: {safe_filename}")
                    raise NotFound()

                return send_from_directory(TEMP_DIR, safe_filename, as_attachment=True)
            except FileNotFoundError:
                logging.error(f"Result file {safe_filename} not found in {TEMP_DIR} for task {task_id}.")
                flash('Result file not found. It might have been cleaned up or failed to save.', 'error')
                return redirect(url_for('main.index'))
            except Exception as e:
                logging.error(f"Error serving file {safe_filename} for task {task_id}: {e}", exc_info=True)
                flash('Error serving result file.', 'error')
                return redirect(url_for('main.index'))
        else:
            logging.error(f"Task {task_id} succeeded but no result filename found in task info.")
            flash('Task completed but result file information is missing.', 'error')
            return redirect(url_for('main.index'))

    elif task.state == 'FAILURE':
        flash('Redaction task failed. Cannot download result.', 'error')
        return redirect(url_for('main.index'))
    else:
        # Task not finished, redirect back to progress page
        flash('Redaction is still in progress. Please wait for completion to download.', 'info')
        # Important: Redirect to the progress page for the specific task, not just index
        return redirect(url_for('main.progress', task_id=task_id))


@bp.route('/progress/<task_id>')
def progress(task_id):
    """Display progress page for a specific task."""
    return render_template('progress.html', task_id=task_id)


# File Serving Routes (Serve files from upload/temp, Use with caution)
@bp.route('/uploads/<path:filename>')
def uploaded_file(filename):
    # Add auth checks if needed
    safe_filename = secure_filename(filename)
    if not safe_filename:
        raise NotFound()
    # Basic path traversal check
    full_path = os.path.join(UPLOAD_DIR, safe_filename)
    if not os.path.abspath(full_path).startswith(os.path.abspath(UPLOAD_DIR)):
        raise NotFound()
    logging.debug(f"Request for uploaded file: {safe_filename}")
    return send_from_directory(UPLOAD_DIR, safe_filename, as_attachment=False)  # Allow viewing?


@bp.route('/temp/<path:filename>')
def temp_file(filename):
    # Add auth checks if needed
    safe_filename = secure_filename(filename)
    if not safe_filename:
        raise NotFound()
    # Basic path traversal check
    full_path = os.path.join(TEMP_DIR, safe_filename)
    if not os.path.abspath(full_path).startswith(os.path.abspath(TEMP_DIR)):
        raise NotFound()
    logging.debug(f"Request for temp file: {safe_filename}")
    # Typically force download for temp files
    return send_from_directory(TEMP_DIR, safe_filename, as_attachment=True)


# Cleanup Endpoint
@bp.route('/cleanup', methods=['POST'])
def trigger_cleanup():
    # !! IMPORTANT: Secure this endpoint in production !!
    form = UploadForm()  # Use form for CSRF token validation
    # This relies on the CSRF token being included in the POST request,
    # which the form in index.html should do.
    if form.validate_on_submit():
        logging.info("Manual cleanup triggered via /cleanup endpoint.")
        try:
            cleaned_count = cleanup_temp_files()
            flash(f"Cleanup process completed. Removed {cleaned_count} old files.", "success")
        except Exception as e:
            logging.error(f"Error during manual cleanup: {e}", exc_info=True)
            flash(f"Cleanup failed: {e}", "error")
    else:
        # Log CSRF error or other potential form validation errors
        csrf_error = form.csrf_token.errors if hasattr(form, 'csrf_token') else []
        logging.warning(f"CSRF validation failed or other error on cleanup POST. Errors: {csrf_error} / {form.errors}")
        flash("Could not validate cleanup request.", "error")
    return redirect(url_for('main.index'))


# Register error handlers
def register_error_handlers(app):
    """Register error handlers with the Flask app."""
    
    @app.errorhandler(404)
    def not_found_error(error):
        logging.warning(f"404 Not Found error: {request.url}")
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        # Log the actual error trace carefully
        logging.error(f"500 Internal Server Error on {request.url}", exc_info=True)
        # Check for specific connection errors to provide better user feedback
        user_message = "An unexpected internal error occurred. Please try again later or contact support."
        original_exception_str = str(getattr(error, 'original_exception', error)).lower()
        if "redis" in original_exception_str or "connection refused" in original_exception_str:
            user_message = "Could not connect to the background task service (Redis). Please ensure it's running and accessible."
        # Render the styled 500 page
        return render_template('500.html', error_message=user_message), 500

    @app.errorhandler(413)
    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(e):
        # Log appropriately
        limit_mb = MAX_FILE_SIZE_MB
        content_length = request.content_length or 'Unknown'
        logging.warning(f"File upload rejected (too large): Request Content-Length {content_length} bytes. Limit: {limit_mb} MB.")
        user_message = f"File exceeds the maximum allowed size of {limit_mb} MB."
        # Return JSON for AJAX requests, redirect for normal form posts
        # Check if the request likely came from our AJAX script
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or \
           (request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html):
            return jsonify({'error': user_message}), 413
        # Otherwise, assume it might be a non-JS submission or fallback
        flash(user_message, 'error')
        return redirect(url_for('main.index'))