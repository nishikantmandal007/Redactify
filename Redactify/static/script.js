document.addEventListener('DOMContentLoaded', function () {

    // --- DOM Elements ---
    const uploadForm = document.getElementById('upload-form');
    const submitButton = document.getElementById('submit-button');
    const submitText = document.getElementById('submit-text');
    const loadingSpinner = document.getElementById('loading-spinner');
    const fileInput = document.getElementById('pdf_file'); // Ensure your FileField has id="pdf_file"
    const selectedFilenameSpan = document.getElementById('selected-filename');

    const inputZone = document.getElementById('input-zone');
    const progressZone = document.getElementById('progress-zone');
    const statusMessageDiv = document.getElementById('status-message');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const resultLinkDiv = document.getElementById('result-link');
    const downloadButton = document.getElementById('download-button');
    const resetButton = document.getElementById('reset-button'); // Add this button to your HTML if needed
    const csrfTokenInput = document.querySelector('input[name="csrf_token"]'); // Get CSRF token

    let pollIntervalId = null; // To store the interval ID for polling

    // --- Event Listeners ---

    // Update filename display on file selection
    if (fileInput && selectedFilenameSpan) {
        fileInput.addEventListener('change', function () {
            selectedFilenameSpan.textContent = this.files.length > 0 ? this.files[0].name : 'No file chosen';
        });
    }

    // Form submission handler
    if (uploadForm) {
        uploadForm.addEventListener('submit', async function (event) {
            event.preventDefault(); // Prevent default page reload
            clearStatus(); // Clear previous statuses
            hideElement(resultLinkDiv); // Hide previous result link

            // Show loading state
            showLoading(true);
            statusMessageDiv.textContent = 'Uploading and queuing task...';
            statusMessageDiv.className = 'alert alert-info'; // Set initial class
            showElement(statusMessageDiv); // Make sure it's visible if previously hidden

            const formData = new FormData(uploadForm);
            const csrfToken = csrfTokenInput ? csrfTokenInput.value : null; // Get CSRF token value

            try {
                const response = await fetch('/process_pdf', {
                    method: 'POST',
                    body: formData,
                    headers: {
                        // Include CSRF token if your app uses it (Flask-WTF usually does)
                        'X-CSRFToken': csrfToken
                    }
                });

                const data = await response.json();

                if (!response.ok) {
                    // Handle errors returned from the backend (e.g., validation, save errors)
                    throw new Error(data.error || `Server error: ${response.statusText}`);
                }

                // Task queued successfully
                hideElement(inputZone);
                showElement(progressZone);
                statusMessageDiv.textContent = 'Task queued successfully. Starting processing...';
                startPolling(data.task_id);

            } catch (error) {
                console.error('Form submission error:', error);
                showError(`Submission failed: ${error.message}`);
                showLoading(false); // Reset button state
            }
        });
    }

    // Reset button handler (optional)
    if (resetButton) {
        resetButton.addEventListener('click', resetUI);
    }


    // --- Helper Functions ---

    function startPolling(taskId) {
        // Clear any existing interval
        if (pollIntervalId) {
            clearInterval(pollIntervalId);
        }
        // Initial check
        checkTaskStatus(taskId);
        // Start polling every 2 seconds
        pollIntervalId = setInterval(() => checkTaskStatus(taskId), 2000);
    }

    async function checkTaskStatus(taskId) {
        try {
            const response = await fetch(`/task_status/${taskId}`);
            if (!response.ok) {
                // Handle server error during status check (e.g., 500)
                throw new Error(`Status check failed: ${response.statusText}`);
            }
            const data = await response.json();

            // Update UI based on task data
            updateProgressUI(data);

            // Stop polling if task is finished (Success or Failure)
            if (data.state === 'SUCCESS' || data.state === 'FAILURE') {
                clearInterval(pollIntervalId);
                pollIntervalId = null; // Clear interval ID
                showLoading(false); // Ensure button is reset if somehow still loading

                if (data.state === 'SUCCESS' && data.result) {
                    // Set download link and show it
                    downloadButton.href = `/result/${taskId}`; // Use task ID for result lookup
                    showElement(resultLinkDiv);
                } else if (data.state === 'SUCCESS' && !data.result) {
                    // Handle success but missing result file case
                    showError("Task completed, but the result file seems to be missing.");
                }
            }

        } catch (error) {
            console.error('Error fetching task status:', error);
            showError('Could not fetch task status. Please check console or try refreshing.');
            // Consider stopping polling on repeated errors?
            // clearInterval(pollIntervalId);
            // pollIntervalId = null;
            // showLoading(false);
        }
    }

    function updateProgressUI(data) {
        const progressPercent = Math.min(Math.max(data.progress || 0, 0), 100); // Clamp between 0-100

        // Update progress bar
        progressBar.style.width = progressPercent + '%';
        progressBar.setAttribute('aria-valuenow', progressPercent);
        progressText.textContent = progressPercent + '%';

        // Update status message and style
        statusMessageDiv.textContent = data.status || 'Waiting...';
        statusMessageDiv.className = 'alert'; // Reset classes
        if (data.state === 'SUCCESS') {
            statusMessageDiv.classList.add('alert-success');
            progressBar.classList.remove('progress-bar-animated', 'progress-bar-striped');
            progressBar.classList.add('bg-success');
        } else if (data.state === 'FAILURE') {
            statusMessageDiv.classList.add('alert-danger');
            progressBar.classList.remove('progress-bar-animated', 'progress-bar-striped');
            progressBar.classList.add('bg-danger');
        } else if (data.state === 'RETRY') {
            statusMessageDiv.classList.add('alert-warning');
            progressBar.classList.add('progress-bar-animated', 'progress-bar-striped');
            progressBar.classList.remove('bg-success', 'bg-danger'); // Ensure correct color during retry
        } else { // PENDING, STARTED, PROGRESS
            statusMessageDiv.classList.add('alert-info');
            progressBar.classList.add('progress-bar-animated', 'progress-bar-striped');
            progressBar.classList.remove('bg-success', 'bg-danger');
        }
        showElement(statusMessageDiv); // Ensure it's visible
    }

    function showLoading(isLoading) {
        if (isLoading) {
            submitButton.disabled = true;
            hideElement(submitText);
            showElement(loadingSpinner);
        } else {
            submitButton.disabled = false;
            showElement(submitText);
            hideElement(loadingSpinner);
        }
    }

    function showError(message) {
        statusMessageDiv.textContent = message;
        statusMessageDiv.className = 'alert alert-danger';
        showElement(statusMessageDiv);
        hideElement(progressZone); // Hide progress bar on error maybe? Or show failed bar?
        showElement(inputZone); // Show input form again
    }

    function clearStatus() {
        hideElement(statusMessageDiv);
        statusMessageDiv.textContent = '';
        statusMessageDiv.className = 'alert'; // Reset class
        progressBar.style.width = '0%';
        progressBar.setAttribute('aria-valuenow', 0);
        progressText.textContent = '0%';
        progressBar.classList.remove('bg-success', 'bg-danger');
        progressBar.classList.add('progress-bar-animated', 'progress-bar-striped');
    }

    // Function to reset the entire UI to initial state
    function resetUI() {
        if (pollIntervalId) {
            clearInterval(pollIntervalId);
            pollIntervalId = null;
        }
        uploadForm.reset(); // Reset form fields
        selectedFilenameSpan.textContent = 'No file chosen'; // Reset filename display
        clearStatus();
        hideElement(progressZone);
        hideElement(resultLinkDiv);
        showElement(inputZone);
        showLoading(false);
    }


    // --- Utility Functions ---
    function showElement(element) {
        if (element) element.style.display = 'block';
    }

    function hideElement(element) {
        if (element) element.style.display = 'none';
    }

});