/**
 * Camera Module JavaScript
 * Uses WebRTC getUserMedia to stream live video, switch front/rear cameras,
 * and capture high-resolution leaf snapshots directly into the analysis pipeline.
 */

(function() {
    let currentStream = null;
    let currentFacingMode = 'environment'; // Prefer rear camera on smartphones for leaves

    const startCameraBtn = document.getElementById('startCameraBtn');
    const stopCameraBtn = document.getElementById('stopCameraBtn');
    const switchCameraBtn = document.getElementById('switchCameraBtn');
    const captureBtn = document.getElementById('captureBtn');

    const cameraVideo = document.getElementById('cameraVideo');
    const cameraCanvas = document.getElementById('cameraCanvas');
    const cameraPlaceholder = document.getElementById('cameraPlaceholder');
    const cameraControls = document.getElementById('cameraControls');
    const cameraGuide = document.getElementById('cameraGuide');

    // Start Video Stream
    async function startCamera(facing = 'environment') {
        // Stop any active streams first
        stopCamera();

        const constraints = {
            video: {
                facingMode: { ideal: facing },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        };

        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Your browser does not support camera capture via WebRTC.');
            }

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            currentStream = stream;
            currentFacingMode = facing;

            if (cameraVideo) {
                cameraVideo.srcObject = stream;
                cameraVideo.style.display = 'block';
                await cameraVideo.play();
            }

            if (cameraPlaceholder) cameraPlaceholder.style.display = 'none';
            if (cameraControls) cameraControls.style.display = 'flex';
            if (cameraGuide) cameraGuide.style.display = 'block';

        } catch (err) {
            console.error('Camera stream error:', err);
            let msg = 'Could not access device camera. Please ensure permissions are granted.';
            if (err.name === 'NotAllowedError') {
                msg = 'Camera access was denied by your browser. Please allow camera permissions in your browser bar.';
            } else if (err.name === 'NotFoundError') {
                msg = 'No camera found on this device.';
            }
            alert(msg);
        }
    }

    // Stop Video Stream
    function stopCamera() {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
        }

        if (cameraVideo) {
            cameraVideo.pause();
            cameraVideo.srcObject = null;
            cameraVideo.style.display = 'none';
        }

        if (cameraPlaceholder) cameraPlaceholder.style.display = 'flex';
        if (cameraControls) cameraControls.style.display = 'none';
        if (cameraGuide) cameraGuide.style.display = 'none';
    }

    // Capture Snapshot
    function captureSnapshot() {
        if (!cameraVideo || !cameraCanvas || !currentStream) {
            alert('Camera is not active.');
            return;
        }

        const videoWidth = cameraVideo.videoWidth || 640;
        const videoHeight = cameraVideo.videoHeight || 480;

        cameraCanvas.width = videoWidth;
        cameraCanvas.height = videoHeight;

        const ctx = cameraCanvas.getContext('2d');
        ctx.drawImage(cameraVideo, 0, 0, videoWidth, videoHeight);

        // Convert to JPEG data URL
        const dataUrl = cameraCanvas.toDataURL('image/jpeg', 0.92);

        // Calculate size in KB
        const head = 'data:image/jpeg;base64,';
        const fileSizeBytes = Math.round((dataUrl.length - head.length) * 3 / 4);
        const sizeKB = (fileSizeBytes / 1024).toFixed(1) + ' KB';
        const timestamp = new Date().toISOString().replace(/[-:T.]/g, '').slice(0, 14);
        const filename = `camera_leaf_${timestamp}.jpg`;

        // Update Global State
        window.selectedLeafDataUrl = dataUrl;
        window.selectedLeafFile = null;
        window.selectedSource = 'camera';

        // Render in preview sidebar
        if (typeof window.renderLeafPreview === 'function') {
            window.renderLeafPreview(null, dataUrl, filename, sizeKB, 'JPEG', 'Live Camera Snapshot');
        }

        // Stop camera feed after snapshot
        stopCamera();
    }

    // Switch Lens (Front vs Rear)
    function switchCamera() {
        const nextMode = currentFacingMode === 'environment' ? 'user' : 'environment';
        startCamera(nextMode);
    }

    // Attach Event Listeners
    if (startCameraBtn) startCameraBtn.addEventListener('click', () => startCamera(currentFacingMode));
    if (stopCameraBtn) stopCameraBtn.addEventListener('click', stopCamera);
    if (switchCameraBtn) switchCameraBtn.addEventListener('click', switchCamera);
    if (captureBtn) captureBtn.addEventListener('click', captureSnapshot);

    // Stop camera when user navigates away or unloads page
    window.addEventListener('beforeunload', stopCamera);
})();
