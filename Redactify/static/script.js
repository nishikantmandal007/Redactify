document.addEventListener('DOMContentLoaded', function () {

    // --- DOM Elements ---
    const uploadForm = document.getElementById('upload-form');
    const submitButton = document.getElementById('submit-button');
    const submitText = document.getElementById('submit-text');
    const loadingSpinner = document.getElementById('loading-spinner');
    const fileInput = document.getElementById('file');
    const selectedFilenameSpan = document.getElementById('selected-filename');
    const fileInputWrapper = document.querySelector('.input-group');

    // --- Advanced Options Toggle ---
    const advancedOptionsToggle = document.getElementById('advanced-options-toggle');
    const advancedOptionsSection = document.getElementById('advanced-options-section');

    // Metadata redaction element
    const redactMetadataCheckbox = document.getElementById('redact_metadata');

    if (advancedOptionsToggle && advancedOptionsSection) {
        advancedOptionsToggle.addEventListener('click', function () {
            // Toggle icon rotation
            const icon = advancedOptionsToggle.querySelector('i');
            if (icon) {
                icon.style.transform = icon.style.transform === 'rotate(180deg)' ? '' : 'rotate(180deg)';
            }

            // Toggle section visibility using Bootstrap classes
            advancedOptionsSection.classList.toggle('d-none');

            // Update button text
            const toggleText = advancedOptionsToggle.querySelector('span');
            if (toggleText) {
                toggleText.textContent = advancedOptionsSection.classList.contains('d-none') ?
                    'Show Advanced Options' : 'Hide Advanced Options';
            }

            // Force redraw of the section (helps with display issues)
            if (!advancedOptionsSection.classList.contains('d-none')) {
                advancedOptionsSection.style.opacity = '0.99';
                setTimeout(() => {
                    advancedOptionsSection.style.opacity = '1';
                }, 0);
            }
        });
    }

    // --- Barcode Options Toggle ---
    const redactBarcodesCheckbox = document.getElementById('redact_barcodes');
    const barcodeOptions = document.getElementById('barcode-options');

    if (redactBarcodesCheckbox && barcodeOptions) {
        function updateBarcodeOptionsVisibility() {
            barcodeOptions.classList.toggle('d-none', !redactBarcodesCheckbox.checked);
        }

        // Initial state
        updateBarcodeOptionsVisibility();

        // Add event listener
        redactBarcodesCheckbox.addEventListener('change', updateBarcodeOptionsVisibility);
    }

    // --- Improve file input handling ---
    if (fileInputWrapper && fileInput) {
        // Update filename display
        fileInput.addEventListener('change', function () {
            const file = this.files.length > 0 ? this.files[0] : null;
            selectedFilenameSpan.textContent = file ? file.name : 'No file chosen';

            // Detect file type for later use
            if (file) {
                const fileName = file.name.toLowerCase();
                if (fileName.endsWith('.pdf')) {
                    currentFileType = 'pdf';
                } else if (fileName.match(/\.(jpe?g|png|gif|bmp|tiff?)$/)) {
                    currentFileType = 'image';
                } else {
                    currentFileType = null;
                    // Alert user about invalid file type
                    statusMessageDiv.className = 'alert alert-warning';
                    statusMessageDiv.innerHTML = '<i class="material-icons-round me-2">warning</i> Please select a PDF or image file (.pdf, .jpg, .jpeg, .png, .gif, .bmp, .tiff)';
                    showElement(statusMessageDiv);
                }
            } else {
                currentFileType = null;
            }
        });
    }

    const inputZone = document.getElementById('input-zone');
    const progressZone = document.getElementById('progress-zone');
    const statusMessageDiv = document.getElementById('status-message');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const resultLinkDiv = document.getElementById('result-link');
    const downloadButton = document.getElementById('download-button');
    const resetButton = document.getElementById('reset-button');
    const csrfTokenInput = document.querySelector('input[name="csrf_token"]'); // Get CSRF token

    // Preview elements
    const togglePreviewBtn = document.getElementById('toggle-preview-btn');
    const pdfPreviewDiv = document.getElementById('pdf-preview');
    const pdfPreviewFrame = document.getElementById('pdf-preview-frame');
    const imagePreviewFrame = document.getElementById('image-preview-frame');
    const previewHeaderText = document.getElementById('preview-header-text'); // For updating header text

    let pollIntervalId = null; // To store the interval ID for polling
    let currentFileType = null; // Store the detected file type: 'pdf' or 'image'
    let isProcessing = false;
    let currentController = null; // AbortController for fetch requests

    // --- Event Listeners ---

    // Toggle file preview
    if (togglePreviewBtn) {
        togglePreviewBtn.addEventListener('click', function () {
            const isHidden = pdfPreviewDiv.classList.contains('d-none');

            if (isHidden) {
                // Show preview
                pdfPreviewDiv.classList.remove('d-none');
                togglePreviewBtn.querySelector('span').textContent = 'Hide Preview';
                togglePreviewBtn.querySelector('i').textContent = 'visibility_off';

                // Load file into iframe if not already loaded
                if (!pdfPreviewFrame.src && downloadButton.href) {
                    pdfPreviewFrame.src = downloadButton.href;
                }
            } else {
                // Hide preview
                pdfPreviewDiv.classList.add('d-none');
                togglePreviewBtn.querySelector('span').textContent = 'Preview File';
                togglePreviewBtn.querySelector('i').textContent = 'visibility';
            }
        });
    }

    // Form submission handler
    if (uploadForm) {
        uploadForm.addEventListener('submit', async function (event) {
            event.preventDefault(); // IMPORTANT: Move to very first line to prevent any chance of normal form submission
            console.log('Form submitted!'); // DEBUG: log every submit event

            // Abort any previous request that might still be in progress
            if (currentController) {
                currentController.abort();
            }

            if (isProcessing) return;
            isProcessing = true;
            submitButton.disabled = true;
            clearStatus(); // Clear previous statuses
            hideElement(resultLinkDiv); // Hide previous result link

            // Show loading state
            showLoading(true);
            statusMessageDiv.innerHTML = '<i class="material-icons-round me-2">cloud_upload</i> Uploading and queuing task...';
            statusMessageDiv.className = 'alert alert-info'; // Set initial class
            showElement(statusMessageDiv); // Make sure it's visible if previously hidden

            const formData = new FormData(uploadForm);
            const csrfToken = csrfTokenInput ? csrfTokenInput.value : null; // Get CSRF token value

            // Create a new AbortController for this request
            currentController = new AbortController();
            const signal = currentController.signal;

            try {
                // Updated to use the same URL as the form action
                const response = await fetch(uploadForm.action, {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'X-CSRFToken': csrfToken, // Include CSRF token in header if needed
                        'X-Requested-By': 'fetch', // Add custom header to identify programmatic requests
                    },
                    signal, // Add the abort signal
                    // Don't set Content-Type header; fetch sets it with boundary for FormData
                });

                // Special handling for 429 Too Many Requests (duplicate submission)
                if (response.status === 429) {
                    console.log('Duplicate request detected, waiting for the other request to complete...');
                    // Don't show an error, just wait for the successful request to complete
                    return;
                }

                if (!response.ok) {
                    throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
                }

                const responseData = await response.json();

                if (responseData.error) {
                    // Handle error from server
                    statusMessageDiv.className = 'alert alert-danger';
                    statusMessageDiv.innerHTML = `<i class="material-icons-round me-2">error</i> ${responseData.error}`;
                    showLoading(false);
                } else if (responseData.task_id) {
                    // Success - hide input, show progress
                    hideElement(inputZone);
                    showElement(progressZone);

                    // Update progress zone title based on file type
                    const progressTitle = document.querySelector('#progress-zone h3');
                    if (progressTitle) {
                        progressTitle.textContent = currentFileType === 'image' ?
                            'Processing Image...' : 'Processing PDF...';
                    }

                    startTaskPolling(responseData.task_id);
                } else {
                    // Unexpected response format
                    statusMessageDiv.className = 'alert alert-warning';
                    statusMessageDiv.innerHTML = '<i class="material-icons-round me-2">warning</i> Unexpected server response. Please try again.';
                    showLoading(false);
                }
            } catch (error) {
                // Network or other error
                console.error('Form submission error:', error);
                statusMessageDiv.className = 'alert alert-danger';
                statusMessageDiv.innerHTML = `<i class="material-icons-round me-2">error</i> Error: ${error.message}`;
                showLoading(false);
            } finally {
                submitButton.disabled = false;
                isProcessing = false;
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
            clearInterval(pollIntervalId);
        }
        // Start polling every 1 second
        pollIntervalId = setInterval(() => {
            fetch(`/task_status/${taskId}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`Server responded with status ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    updateTaskProgress(data, taskId);
                })
                .catch(error => {
                    console.error('Polling error:', error);
                    clearInterval(pollIntervalId);
                    pollIntervalId = null;
                    statusMessageDiv.className = 'alert alert-danger';
                    statusMessageDiv.innerHTML = `<i class="material-icons-round me-2">error</i> Error checking task status: ${error.message}`;
                });
        }, 1000); // Poll every 1 second
    }

    function updateTaskProgress(data, taskId) {
        // Update progress bar
        if (progressBar && progressText) {
            progressBar.style.width = `${data.progress}%`;
            progressBar.setAttribute('aria-valuenow', data.progress);
            progressText.textContent = `${data.progress}%`;
        }

        // Update status message text (preserve icon)
        if (statusMessageDiv) {
            const existingIcon = statusMessageDiv.querySelector('i')?.outerHTML || '';
            statusMessageDiv.innerHTML = existingIcon + ' ' + data.status;
        }

        // Based on state, update UI
        if (data.state === 'SUCCESS') {
            clearInterval(pollIntervalId);
            pollIntervalId = null;

            // Complete the progress bar
            if (progressBar) {
                progressBar.style.width = '100%';
                progressBar.setAttribute('aria-valuenow', 100);
                progressBar.classList.remove('progress-bar-animated', 'progress-bar-striped');
                progressBar.classList.add('bg-success');
            }

            if (progressText) progressText.textContent = '100%';

            // Show success message with check icon
            if (statusMessageDiv) {
                statusMessageDiv.className = 'alert alert-success';
                statusMessageDiv.innerHTML = `<i class="material-icons-round me-2">check_circle</i> ${data.status}`;

                // Add metadata redaction info if available
                if (data.result && data.metadata_stats && data.metadata_stats.cleaned) {
                    const metadataInfo = document.createElement('div');
                    metadataInfo.className = 'mt-2 p-2 bg-light rounded';
                    metadataInfo.innerHTML = `
                        <strong>Document Metadata Cleaning Results:</strong>
                        <ul class="mb-0 small">
                            <li>Document properties cleaned (author, title, etc.)</li>
                            ${data.metadata_stats.hidden_text_removed ? '<li>Hidden text layers removed</li>' : ''}
                            ${data.metadata_stats.embedded_files_removed ? '<li>Embedded files and JavaScript removed</li>' : ''}
                            ${data.metadata_stats.history_cleaned ? '<li>Document revision history cleaned</li>' : ''}
                        </ul>
                    `;
                    statusMessageDiv.appendChild(metadataInfo);
                }
            }

            // Show download link if result filename is provided
            if (data.result) {
                // Update download button & preview
                if (downloadButton) {
                    downloadButton.href = `/result/${taskId}`;
                    showElement(resultLinkDiv);
                }

                // Update preview section based on file type
                const fileExt = data.result.toLowerCase();
                const isImage = fileExt.match(/\.(jpe?g|png|gif|bmp|tiff?)$/);
                const isPdf = fileExt.endsWith('.pdf');

                if (pdfPreviewFrame && isPdf) {
                    pdfPreviewFrame.classList.remove('d-none');
                    pdfPreviewFrame.src = `/preview/${taskId}`;

                    if (imagePreviewFrame) {
                        imagePreviewFrame.classList.add('d-none');
                    }
                } else if (imagePreviewFrame && isImage) {
                    imagePreviewFrame.classList.remove('d-none');
                    imagePreviewFrame.src = `/preview/${taskId}`;

                    if (pdfPreviewFrame) {
                        pdfPreviewFrame.classList.add('d-none');
                    }
                }
            }
        } else if (data.state === 'FAILURE') {
            clearInterval(pollIntervalId);
            pollIntervalId = null;
            // Show error UI
            if (progressBar) {
                progressBar.classList.remove('progress-bar-animated', 'progress-bar-striped');
                progressBar.classList.add('bg-danger');
            }
            // Show error message
            if (statusMessageDiv) {
                statusMessageDiv.className = 'alert alert-danger';
                statusMessageDiv.innerHTML = `<i class="material-icons-round me-2">error</i> ${data.status}`;
            }
        }
        // Otherwise keep polling...
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
        currentFileType = null;

        // Reset preview
        if (pdfPreviewDiv) {
            pdfPreviewDiv.classList.add('d-none');
        }
        if (togglePreviewBtn) {
            togglePreviewBtn.querySelector('span').textContent = 'Preview File';
            togglePreviewBtn.querySelector('i').textContent = 'visibility';
        }
        if (pdfPreviewFrame) {
            pdfPreviewFrame.src = '';
            pdfPreviewFrame.classList.add('d-none');
        }
        if (imagePreviewFrame) {
            imagePreviewFrame.src = '';
            imagePreviewFrame.classList.add('d-none');
        }

        // Reset advanced options if open
        if (advancedOptionsToggle && advancedOptionsSection) {
            if (!advancedOptionsSection.classList.contains('d-none')) {
                advancedOptionsSection.classList.add('d-none');
                const icon = advancedOptionsToggle.querySelector('i');
                if (icon) icon.style.transform = '';

                const toggleText = advancedOptionsToggle.querySelector('span');
                if (toggleText) toggleText.textContent = 'Show Advanced Options';
            }
        }
    }

    // --- Utility Functions ---
    function showElement(element) {
        if (element) {
            element.classList.remove('d-none');
        }
    }

    function hideElement(element) {
        if (element) {
            element.classList.add('d-none');
        }
    }
});