/**
 * Disease Detection Interface Logic
 * Handles file drag-and-drop, client-side validation, previewing, and API submission.
 */

// Selected Image State
window.selectedLeafFile = null;
window.selectedLeafDataUrl = null;
window.selectedSource = 'upload'; // 'upload' or 'camera'

document.addEventListener('DOMContentLoaded', () => {
    // Tab switching
    const tabUpload = document.getElementById('tabUpload');
    const tabCamera = document.getElementById('tabCamera');
    const paneUpload = document.getElementById('paneUpload');
    const paneCamera = document.getElementById('paneCamera');

    if (tabUpload && tabCamera && paneUpload && paneCamera) {
        tabUpload.addEventListener('click', () => {
            tabUpload.classList.add('active');
            tabCamera.classList.remove('active');
            paneUpload.classList.add('active');
            paneCamera.classList.remove('active');
        });

        tabCamera.addEventListener('click', () => {
            tabCamera.classList.add('active');
            tabUpload.classList.remove('active');
            paneCamera.classList.add('active');
            paneUpload.classList.remove('active');
        });
    }

    // Drag and drop zone elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');

    // Preview elements
    const previewEmpty = document.getElementById('previewEmpty');
    const previewActive = document.getElementById('previewActive');
    const imagePreview = document.getElementById('imagePreview');
    const previewSourceTag = document.getElementById('previewSourceTag');
    const selectionStatus = document.getElementById('selectionStatus');
    const detailName = document.getElementById('detailName');
    const detailSize = document.getElementById('detailSize');
    const detailFormat = document.getElementById('detailFormat');

    // Controls
    const changeImageBtn = document.getElementById('changeImageBtn');
    const removeImageBtn = document.getElementById('removeImageBtn');
    const analyseLeafBtn = document.getElementById('analyseLeafBtn');
    const btnSpinner = document.getElementById('btnSpinner');
    const btnIcon = document.getElementById('btnIcon');
    const btnText = document.getElementById('btnText');

    // Error & Outcome
    const validationError = document.getElementById('validationError');
    const validationErrorText = document.getElementById('validationErrorText');
    const analysisOutcome = document.getElementById('analysisOutcome');
    const outcomeTitle = document.getElementById('outcomeTitle');
    const outcomeMessage = document.getElementById('outcomeMessage');
    const viewResultsLayoutBtn = document.getElementById('viewResultsLayoutBtn');

    // Supported formats and constraints
    const ALLOWED_MIME = ['image/jpeg', 'image/jpg', 'image/png'];
    const ALLOWED_EXT = ['jpg', 'jpeg', 'png'];
    const MAX_SIZE_BYTES = 16 * 1024 * 1024; // 16 MB

    // Show / Hide Error
    function showError(message) {
        if (validationError && validationErrorText) {
            validationErrorText.textContent = message;
            validationError.style.display = 'flex';
        }
    }

    function clearError() {
        if (validationError) {
            validationError.style.display = 'none';
        }
    }

    // Validate Selected File
    function validateFile(file) {
        clearError();

        if (!file) {
            showError('Please select a strawberry leaf photograph.');
            return false;
        }

        // Check extension
        const ext = file.name.split('.').pop().toLowerCase();
        if (!ALLOWED_EXT.includes(ext) && !ALLOWED_MIME.includes(file.type)) {
            showError(`Invalid file format (.${ext}). Only JPG, JPEG, and PNG images are supported.`);
            return false;
        }

        // Check size limit
        if (file.size > MAX_SIZE_BYTES) {
            const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
            showError(`File size (${sizeMB} MB) exceeds the 16 MB limit. Please upload a smaller image.`);
            return false;
        }

        return true;
    }

    // Render Preview
    window.renderLeafPreview = function(fileOrBlob, dataUrl, filename, fileSize, format, sourceTag) {
        clearError();
        if (analysisOutcome) analysisOutcome.style.display = 'none';

        if (imagePreview) imagePreview.src = dataUrl;
        if (previewSourceTag) previewSourceTag.textContent = sourceTag || 'Uploaded Image';
        if (selectionStatus) selectionStatus.textContent = 'Image Ready';
        if (detailName) detailName.textContent = filename || 'leaf_capture.png';
        if (detailSize) detailSize.textContent = fileSize || 'Unknown';
        if (detailFormat) detailFormat.textContent = format || 'image/png';

        if (previewEmpty) previewEmpty.style.display = 'none';
        if (previewActive) previewActive.style.display = 'block';
    };

    // Reset Preview
    window.resetLeafPreview = function() {
        window.selectedLeafFile = null;
        window.selectedLeafDataUrl = null;
        if (fileInput) fileInput.value = '';

        if (previewEmpty) previewEmpty.style.display = 'block';
        if (previewActive) previewActive.style.display = 'none';
        if (analysisOutcome) analysisOutcome.style.display = 'none';
        if (selectionStatus) selectionStatus.textContent = 'No Image Loaded';
        clearError();
    };

    // Process Selected File Object
    function handleFileSelection(file) {
        if (!validateFile(file)) {
            resetLeafPreview();
            return;
        }

        window.selectedLeafFile = file;
        window.selectedSource = 'upload';

        const reader = new FileReader();
        reader.onload = (e) => {
            window.selectedLeafDataUrl = e.target.result;
            const sizeKB = (file.size / 1024).toFixed(1) + ' KB';
            const ext = file.name.split('.').pop().toUpperCase();
            window.renderLeafPreview(file, e.target.result, file.name, sizeKB, ext, 'Uploaded Image');
        };
        reader.readAsDataURL(file);
    }

    // Browse Button
    if (browseBtn && fileInput) {
        browseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });

        // Drag and Drop listeners
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('dragover');
            });
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt && dt.files && dt.files.length > 0) {
                handleFileSelection(dt.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelection(e.target.files[0]);
            }
        });
    }

    // Change / Remove Buttons
    if (changeImageBtn && fileInput) {
        changeImageBtn.addEventListener('click', () => {
            fileInput.click();
        });
    }

    if (removeImageBtn) {
        removeImageBtn.addEventListener('click', () => {
            resetLeafPreview();
        });
    }

    // Submit Analysis to Backend (/api/detect)
    if (analyseLeafBtn) {
        analyseLeafBtn.addEventListener('click', async () => {
            if (!window.selectedLeafFile && !window.selectedLeafDataUrl) {
                showError('Please select or capture a strawberry leaf photograph before analyzing.');
                return;
            }

            clearError();

            // Set Loading UI State
            if (btnSpinner) btnSpinner.style.display = 'inline-block';
            if (btnIcon) btnIcon.style.display = 'none';
            if (btnText) btnText.textContent = 'Processing Foliage...';
            analyseLeafBtn.disabled = true;

            try {
                let response;

                if (window.selectedSource === 'camera' && window.selectedLeafDataUrl) {
                    // Send Camera Snapshot as JSON base64
                    response = await fetch('/api/detect', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            camera_image: window.selectedLeafDataUrl
                        })
                    });
                } else if (window.selectedLeafFile) {
                    // Send File via FormData
                    const formData = new FormData();
                    formData.append('file', window.selectedLeafFile);

                    response = await fetch('/api/detect', {
                        method: 'POST',
                        body: formData
                    });
                } else {
                    throw new Error('No valid image data available.');
                }

                const result = await response.json();

                if (!response.ok || !result.success) {
                    throw new Error(result.error || 'Server rejected image upload.');
                }

                // Display Clean Outcome Card
                if (analysisOutcome) {
                    analysisOutcome.style.display = 'block';

                    if (outcomeTitle) outcomeTitle.textContent = 'Image Uploaded Successfully';
                    if (outcomeMessage) {
                        outcomeMessage.textContent = result.message || 
                            'Model integration pending. Your image has been uploaded successfully and preprocessed for the ResNet9-SE pipeline.';
                    }

                    // Update Results Layout Link
                    if (viewResultsLayoutBtn && result.filename) {
                        viewResultsLayoutBtn.href = `/results?file=${encodeURIComponent(result.filename)}&class=Leaf%20Scorch`;
                    }

                    // Smooth scroll down to the outcome card
                    analysisOutcome.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }

            } catch (err) {
                showError(`Analysis failed: ${err.message}`);
            } finally {
                // Reset Button UI State
                if (btnSpinner) btnSpinner.style.display = 'none';
                if (btnIcon) btnIcon.style.display = 'inline-block';
                if (btnText) btnText.textContent = 'Analyse Leaf';
                analyseLeafBtn.disabled = false;
            }
        });
    }
});
