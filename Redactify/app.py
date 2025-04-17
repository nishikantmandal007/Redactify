# Redactify/app.py - COMPLETE AND CORRECTED
import os
import shutil
import logging
import time
import re # Import re for regex compilation check

from flask import (
    Flask, render_template, redirect, url_for, flash,
    send_from_directory, request, jsonify, current_app
)
# Removed: from flask_material import Material
from flask_bootstrap import Bootstrap
# Import Celery app instance and AsyncResult
from .celery_app import celery # Import from the new celery_app.py
from celery.result import AsyncResult
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge, NotFound

# --- Use relative imports for local modules ---
from .forms import UploadForm
from .utils import (
    detect_pdf_type, # Keep detect_pdf_type if called directly by task, but task should handle it
    get_pii_types as get_pii_choices_from_util,
    cleanup_temp_files
)
from .config import ( # Import necessary config for Flask app itself
    UPLOAD_DIR, TEMP_DIR, MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB, TEMP_FILE_MAX_AGE_SECONDS
)
# --- Import the task function ---
from .tasks import perform_redaction # Import the task from tasks.py

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - FLASK - %(message)s')

app = Flask(__name__)
Bootstrap(app)

# --- Flask Configuration ---
# Load Secret Key Securely
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'change_this_insecure_default_dev_key_123!') # Use env var
if app.config['SECRET_KEY'] == 'change_this_insecure_default_dev_key_123!':
    logging.warning("SECURITY WARNING: Using default insecure FLASK_SECRET_KEY. Set FLASK_SECRET_KEY environment variable.")

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_BYTES
app.config['UPLOAD_EXTENSIONS'] = ['.pdf'] # Allowed extensions
# Celery related config is now primarily in celery_app.py
# app.config['CELERY_BROKER_URL'] = REDIS_URL # Not strictly needed here anymore
# app.config['CELERY_RESULT_BACKEND'] = REDIS_URL # Not strictly needed here anymore

# --- Helper Function ---
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    has_dot = '.' in filename
    if not has_dot:
        logging.debug(f"allowed_file check: No '.' found in filename '{filename}'")
        return False
    # Ensure TEMP_FOLDER is defined or imported if needed here, but UPLOAD_EXTENSIONS is what matters
    allowed = current_app.config['UPLOAD_EXTENSIONS'] # Use current_app context if needed
    ext = os.path.splitext(filename)[1].lower()
    is_allowed = ext in allowed
    logging.debug(f"allowed_file check: filename='{filename}', extension='{ext}', allowed={allowed}, result={is_allowed}")
    return is_allowed
# --- End Helper Function ---


# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Render the main upload page."""
    form = UploadForm()
    # Dynamically load choices for PII types
    pii_choices = [(ptype, ptype.replace('_', ' ').title()) for ptype in get_pii_choices_from_util()]
    form.pii_types.choices = pii_choices
    return render_template('index.html', form=form, max_size_mb=MAX_FILE_SIZE_MB, TEMP_FILE_MAX_AGE_SECONDS=TEMP_FILE_MAX_AGE_SECONDS)

@app.route('/process_pdf', methods=['POST'])
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

    # --- Input Validation ---
    # Temporarily bypass allowed_file for debugging - Comment out original line:
    # if not pdf_file or not allowed_file(pdf_file.filename):

    # Add the direct check instead:
    if not pdf_file or not pdf_file.filename.lower().endswith('.pdf'):
        # Keep the logging and return statement the same
        logging.warning(f"AJAX POST /process_pdf: Invalid file type or empty file (Direct Check): {pdf_file.filename}")
        return jsonify({'error': 'Invalid file type. Please upload a PDF.'}), 400
    # --- End Input Validation ---


    # --- Secure Filename and Path ---
    filename = secure_filename(pdf_file.filename)
    if not filename:
         logging.warning("AJAX POST /process_pdf: Invalid secure filename generated.")
         return jsonify({'error': 'Invalid filename.'}), 400

    pdf_path = os.path.join(UPLOAD_DIR, filename) # Use UPLOAD_DIR from config

    # Avoid overwriting - generate unique name if exists
    counter = 1
    base, ext = os.path.splitext(pdf_path)
    while os.path.exists(pdf_path):
        pdf_path = f"{base}_{counter}{ext}"
        filename = os.path.basename(pdf_path) # Update filename if changed
        counter += 1
        if counter > 100: # Safety break
             logging.error("AJAX POST /process_pdf: Could not generate unique filename.")
             return jsonify({'error': 'Could not save file, name conflict.'}), 500

    # --- Try Saving the File ---
    try:
        pdf_file.save(pdf_path)
        logging.info(f"AJAX POST /process_pdf: File saved to {pdf_path}")
    except RequestEntityTooLarge:
        # Caught by error handler, but log here too
        logging.warning(f"AJAX POST /process_pdf: File too large: {filename}")
        # The error handler will return the response
        raise # Re-raise to trigger the handler
    except Exception as e:
        logging.error(f"AJAX POST /process_pdf: Failed to save uploaded file {filename}: {e}", exc_info=True)
        return jsonify({'error': f'Error saving file: {e}'}), 500

    # --- Get Other Form Data ---
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
                      re.compile(rx_strip) # Try compiling regex
                      custom_rules["regex"].append(rx_strip)
                  except re.error as regex_err:
                       logging.warning(f"Invalid regex pattern submitted and ignored: {rx_strip} - Error: {regex_err}")
                       # Decide if you want to fail the request or just ignore invalid regexes
                       # return jsonify({'error': f'Invalid regex pattern: "{rx_strip}"'}), 400 # Option to fail
                       pass # Option to ignore

    # --- Queue Task ---
    try:
        # Use the imported task function
        task = perform_redaction.delay(pdf_path, pii_types_selected, custom_rules)
        logging.info(f"AJAX POST /process_pdf: Task {task.id} queued for file {filename}")
        return jsonify({'task_id': task.id}), 202
    except Exception as e:
        logging.error(f"AJAX POST /process_pdf: Failed to queue task for {filename}: {e}", exc_info=True)
        # Clean up saved file if queuing fails
        if os.path.exists(pdf_path):
            try: os.remove(pdf_path)
            except OSError as del_err: logging.warning(f"Could not delete file {pdf_path} after task queue failure: {del_err}")
        return jsonify({'error': f'Error queueing redaction task. Is the background service running?'}), 500


@app.route('/task_status/<task_id>')
def task_status(task_id):
    """Provides task status updates for AJAX polling."""
    try:
        # Use the imported celery instance implicitly via AsyncResult
        task = AsyncResult(task_id, app=celery) # Pass celery app instance
    except Exception as e:
         logging.error(f"Error creating AsyncResult for task_id {task_id}: {e}")
         return jsonify({'state': 'ERROR', 'status': 'Invalid Task ID format or backend error.'}), 404

    response = {
        'state': task.state,
        'status': 'Waiting...',
        'progress': 0,
        'result': None,
    }

    if not task: # Should not happen if AsyncResult created, but check anyway
        response['state'] = 'ERROR'
        response['status'] = 'Task ID not found or invalid.'
        return jsonify(response), 404

    # --- Update response based on task state ---
    if task.state == 'PENDING':
        response['status'] = 'Task is waiting in queue...'
    elif task.state == 'STARTED':
        info = task.info or {}
        response['status'] = info.get('status', 'Task has started...')
    elif task.state == 'PROGRESS':
        info = task.info or {}
        response['progress'] = int((info.get('current', 0) / info.get('total', 1)) * 100) if info.get('total', 1) > 0 else 0
        response['status'] = info.get('status', 'Processing...')
    elif task.state == 'SUCCESS':
        info = task.info or {}
        response['progress'] = 100
        response['status'] = info.get('status', 'Task completed successfully!')
        response['result'] = info.get('result') # Include result filename
    elif task.state == 'FAILURE':
        info = task.info or {}
        response['progress'] = 100 # Show 100% but indicate failure
        response['status'] = f"Task failed: {info.get('exc_type', 'Error')}. Check server logs for details."
        logging.error(f"Reporting FAILURE for task {task_id}: {info.get('exc_type', '')} - {info.get('exc_message', '')}")
    elif task.state == 'RETRY':
        info = task.info or {}
        try:
            current_retry = task.retries + 1 # retries is 0-indexed
            max_retries = task.max_retries if hasattr(task, 'max_retries') and task.max_retries is not None else 'N/A'
            # Celery doesn't easily expose ETA in info, might need custom state update
            status_suffix = f" (Retry {current_retry}/{max_retries})"
        except Exception:
            status_suffix = " (Retrying...)"
        response['status'] = f"Task encountered temporary issue{status_suffix}. Reason: {info.get('exc_type', 'Error')}"
        # Keep progress from previous attempt if available?
        response['progress'] = info.get('progress_before_retry', 0)
    else: # Handle other potential states (REVOKED, etc.)
        response['status'] = f"Task state: {task.state}"

    return jsonify(response)


@app.route('/result/<task_id>')
def result(task_id):
    """Serves the final redacted PDF file based on task ID."""
    try:
        task = AsyncResult(task_id, app=celery) # Pass celery app instance
    except Exception as e:
         logging.error(f"Error creating AsyncResult for task_id {task_id} in /result: {e}")
         flash("Invalid Task ID format.", "error")
         return redirect(url_for('index'))

    if not task:
         flash(f"Unknown task ID: {task_id}", "error")
         return redirect(url_for('index'))

    if task.state == 'SUCCESS':
        result_info = task.info or {}
        redacted_filename = result_info.get('result')
        if redacted_filename:
             safe_filename = secure_filename(redacted_filename)
             if not safe_filename:
                  logging.error(f"Invalid result filename format for task {task_id}: {redacted_filename}")
                  flash("Result filename is invalid.", "error")
                  return redirect(url_for('index'))
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
                 return redirect(url_for('index'))
             except Exception as e:
                 logging.error(f"Error serving file {safe_filename} for task {task_id}: {e}", exc_info=True)
                 flash('Error serving result file.', 'error')
                 return redirect(url_for('index'))
        else:
             logging.error(f"Task {task_id} succeeded but no result filename found in task info.")
             flash('Task completed but result file information is missing.', 'error')
             return redirect(url_for('index'))

    elif task.state == 'FAILURE':
         flash('Redaction task failed. Cannot download result.', 'error')
         return redirect(url_for('index'))
    else:
         # Task not finished, redirect back to progress page
         flash('Redaction is still in progress. Please wait for completion to download.', 'info')
         # Important: Redirect to the *progress* page for the *specific task*, not just index
         return redirect(url_for('progress', task_id=task_id))


# --- File Serving Routes (Serve files from upload/temp, Use with caution) ---
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    # Add auth checks if needed
    safe_filename = secure_filename(filename)
    if not safe_filename: raise NotFound()
    # Basic path traversal check
    full_path = os.path.join(UPLOAD_DIR, safe_filename)
    if not os.path.abspath(full_path).startswith(os.path.abspath(UPLOAD_DIR)):
         raise NotFound()
    logging.debug(f"Request for uploaded file: {safe_filename}")
    return send_from_directory(UPLOAD_DIR, safe_filename, as_attachment=False) # Allow viewing?

@app.route('/temp/<path:filename>')
def temp_file(filename):
    # Add auth checks if needed
    safe_filename = secure_filename(filename)
    if not safe_filename: raise NotFound()
    # Basic path traversal check
    full_path = os.path.join(TEMP_DIR, safe_filename)
    if not os.path.abspath(full_path).startswith(os.path.abspath(TEMP_DIR)):
         raise NotFound()
    logging.debug(f"Request for temp file: {safe_filename}")
    # Typically force download for temp files
    return send_from_directory(TEMP_DIR, safe_filename, as_attachment=True)


# --- Cleanup Endpoint ---
@app.route('/cleanup', methods=['POST'])
def trigger_cleanup():
    # !! IMPORTANT: Secure this endpoint in production !!
    form = UploadForm() # Use form for CSRF token validation
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
    return redirect(url_for('index'))


# --- Error Handlers ---
@app.errorhandler(404)
def not_found_error(error):
    logging.warning(f"404 Not Found error: {request.url}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    # Log the actual error trace carefully
    # Avoid logging sensitive info if possible, rely on detailed logs elsewhere
    logging.error(f"500 Internal Server Error on {request.url}", exc_info=True) # Log full traceback server-side
    # Check for specific connection errors to provide better user feedback
    user_message = "An unexpected internal error occurred. Please try again later or contact support."
    original_exception_str = str(getattr(error, 'original_exception', error)).lower()
    if "redis" in original_exception_str or "connection refused" in original_exception_str:
         user_message = "Could not connect to the background task service (Redis). Please ensure it's running and accessible."
         # No need to flash here, render_template will show it
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
    return redirect(url_for('index'))


# --- Main Execution Block ---
# This block is typically NOT run when using `flask run` or a WSGI server like Gunicorn.
# It's useful for direct execution (`python -m Redactify.app`) but less common now.
if __name__ == '__main__':
    print("*"*50)
    print("Starting Flask development server directly (using app.run)")
    print("This is usually NOT the recommended way for development with Flask CLI.")
    print(f"  Upload Dir: {UPLOAD_DIR}")
    print(f"  Temp Dir: {TEMP_DIR}")
    print(f"  Max File Size: {MAX_FILE_SIZE_MB} MB")
    print("IMPORTANT:")
    print("  1. Ensure Redis server is running.")
    print("  2. Start the Celery worker in a SEPARATE terminal using:")
    # Use app.name which Flask sets
    print(f"     celery -A Redactify.celery_app.celery worker --loglevel=info") # Point to celery_app
    print("*"*50)
    # Enable debug=True ONLY for local debugging, NEVER in production.
    # Use host='0.0.0.0' to make accessible on network (use with caution)
    app.run(debug=True, host='0.0.0.0', port=5000)