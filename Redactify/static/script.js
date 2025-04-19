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

    // PDF Preview elements
    const togglePreviewBtn = document.getElementById('toggle-preview-btn');
    const pdfPreviewDiv = document.getElementById('pdf-preview');
    const pdfPreviewFrame = document.getElementById('pdf-preview-frame');

    let pollIntervalId = null; // To store the interval ID for polling

    // --- Event Listeners ---

    // Update filename display on file selection
    if (fileInput && selectedFilenameSpan) {
        fileInput.addEventListener('change', function () {
            selectedFilenameSpan.textContent = this.files.length > 0 ? this.files[0].name : 'No file chosen';
        });
    }

    // Toggle PDF preview
    if (togglePreviewBtn) {
        togglePreviewBtn.addEventListener('click', function () {
            if (pdfPreviewDiv.style.display === 'none') {
                // Show preview
                pdfPreviewDiv.style.display = 'block';
                togglePreviewBtn.querySelector('span').textContent = 'Hide Preview';
                togglePreviewBtn.querySelector('i').textContent = 'visibility_off';

                // Load PDF into iframe if not already loaded
                if (!pdfPreviewFrame.src) {
                    pdfPreviewFrame.src = downloadButton.href;
                }
            } else {
                // Hide preview
                pdfPreviewDiv.style.display = 'none';
                togglePreviewBtn.querySelector('span').textContent = 'Preview PDF';
                togglePreviewBtn.querySelector('i').textContent = 'visibility';
            }
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
                // Updated to use the same URL as the form action
                const response = await fetch(uploadForm.action, {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'X-CSRFToken': csrfToken, // Include CSRF token in header if needed
                    },
                    // Don't set Content-Type header; fetch sets it with boundary for FormData
                });

                if (!response.ok) {
                    throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
                }

                const responseData = await response.json();

                if (responseData.error) {
                    // Handle error from server
                    statusMessageDiv.className = 'alert alert-danger';
                    statusMessageDiv.textContent = responseData.error;
                    showLoading(false);
                } else if (responseData.task_id) {
                    // Success - hide input, show progress
                    hideElement(inputZone);
                    showElement(progressZone);
                    startTaskPolling(responseData.task_id);
                } else {
                    // Unexpected response format
                    statusMessageDiv.className = 'alert alert-warning';
                    statusMessageDiv.textContent = 'Unexpected server response. Please try again.';
                    showLoading(false);
                }
            } catch (error) {
                // Network or other error
                console.error('Form submission error:', error);
                statusMessageDiv.className = 'alert alert-danger';
                statusMessageDiv.textContent = 'Error: ' + error.message;
                showLoading(false);
            }
        });
    }

    // Reset button handler
    if (resetButton) {
        resetButton.addEventListener('click', resetUI);
    }

    // --- Polling Functions ---
    function startTaskPolling(taskId) {
        if (pollIntervalId) {
            clearInterval(pollIntervalId); // Clear any existing poll
        }

        pollIntervalId = setInterval(async function () {
            try {
                const response = await fetch(`/task_status/${taskId}`);
                if (!response.ok) {
                    throw new Error(`Server responded with ${response.status}`);
                }

                const data = await response.json();
                updateTaskProgress(data, taskId);

                // Check if we should stop polling
                if (data.state === 'SUCCESS' || data.state === 'FAILURE') {
                    clearInterval(pollIntervalId);
                    pollIntervalId = null;
                    showLoading(false);
                }
            } catch (error) {
                console.error('Polling error:', error);
                statusMessageDiv.textContent = 'Error checking task status: ' + error.message;
                statusMessageDiv.className = 'alert alert-danger';
            }
        }, 2000); // Poll every 2 seconds
    }

    function updateTaskProgress(data, taskId) {
        const progressPercent = Math.round(data.progress || 0);

        // Update progress bar
        if (progressBar) {
            progressBar.style.width = `${progressPercent}%`;
            progressBar.setAttribute('aria-valuenow', progressPercent);
            if (progressText) progressText.textContent = `${progressPercent}%`;
        }

        // Update status message
        if (statusMessageDiv && data.status) {
            statusMessageDiv.textContent = data.status;

            // Update alert type based on task state
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
            } else {
                statusMessageDiv.classList.add('alert-info');
                progressBar.classList.add('progress-bar-animated', 'progress-bar-striped');
            }
        }

        // Handle task completion
        if (data.state === 'SUCCESS' && data.result) {
            if (downloadButton) {
                downloadButton.href = `/result/${taskId}`;
                showElement(resultLinkDiv);

                // Automatically load PDF preview when available
                if (pdfPreviewFrame && pdfPreviewDiv) {
                    // Use the preview endpoint instead of temp endpoint for iframe
                    pdfPreviewFrame.src = `/preview/${taskId}`;
                    showElement(pdfPreviewDiv);
                }
            }
        }
    }

    // --- Helper Functions ---

    function clearStatus() {
        if (statusMessageDiv) statusMessageDiv.textContent = '';
        if (progressBar) {
            progressBar.style.width = '0%';
            progressBar.setAttribute('aria-valuenow', 0);
            progressBar.classList.remove('bg-success', 'bg-danger');
            progressBar.classList.add('progress-bar-animated', 'progress-bar-striped');
        }
        if (progressText) progressText.textContent = '0%';
    }

    function showLoading(isLoading) {
        if (submitButton) {
            submitButton.disabled = isLoading;
            loadingSpinner.style.display = isLoading ? 'inline-block' : 'none';
            submitText.style.opacity = isLoading ? '0.7' : '1';
        }
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

        // Reset PDF preview
        if (pdfPreviewDiv) {
            pdfPreviewDiv.style.display = 'none';
        }
        if (togglePreviewBtn) {
            togglePreviewBtn.querySelector('span').textContent = 'Preview PDF';
            togglePreviewBtn.querySelector('i').textContent = 'visibility';
        }
        if (pdfPreviewFrame) {
            pdfPreviewFrame.src = '';
        }
    }


    // --- Utility Functions ---
    function showElement(element) {
        if (element) element.style.display = 'block';
    }

    function hideElement(element) {
        if (element) element.style.display = 'none';
    }

});