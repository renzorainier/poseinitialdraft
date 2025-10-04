import React, { useRef, useEffect, useState, useCallback } from 'react';

// --- HELPER FUNCTION: Colormapping function for Viridis (Blue to Yellow) ---
/**
 * Maps a single 8-bit depth value (0-255) to an RGB color triplet using a simplified Viridis colormap.
 * @param {number} value - The depth value (0 = far, 255 = near).
 * @returns {Array<number>} An array [R, G, B] where each component is 0-255.
 */
function mapValueToViridisColor(value) {
    // Normalize value from [0, 255] to [0, 1]
    const t = value / 255.0;

    let r = 0, g = 0, b = 0;

    // Simplified Viridis Approximation (Far = Blue, Near = Yellow/White)
    if (t < 0.25) {
        r = 0;
        g = Math.floor(180 * t * 4);
        b = 255 - Math.floor(255 * t * 4 * 0.5);
    } else if (t < 0.5) {
        r = Math.floor(255 * (t - 0.25) * 4 * 0.5);
        g = 180 + Math.floor(75 * (t - 0.25) * 4);
        b = 130 - Math.floor(130 * (t - 0.25) * 4);
    } else if (t < 0.75) {
        r = 128 + Math.floor(127 * (t - 0.5) * 4);
        g = 255;
        b = 0;
    } else {
        r = 255;
        g = 255 - Math.floor(255 * (t - 0.75) * 4 * 0.5);
        b = Math.floor(255 * (t - 0.75) * 4 * 0.5);
    }

    return [Math.min(255, r), Math.min(255, g), Math.min(255, b)];
}

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;

function DepthEstimator() {
    const workerRef = useRef(null);
    const [status, setStatus] = useState('Initialize the model and start webcam.');
    const [isCameraActive, setIsCameraActive] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const videoRef = useRef(null);
    const canvasRef = useRef(null); // Canvas for displaying Depth Map
    const captureCanvasRef = useRef(null); // Hidden canvas for frame capture

    // --- WORKER SETUP ---
    useEffect(() => {
        // FIX: Replaced new URL(..., import.meta.url) with a simple path string
        // to resolve the 'empty-import-meta' compilation warning.
        workerRef.current = new Worker('./depth-worker.js', { type: 'module' });

        // Pre-initialize the model to get the heavy loading out of the way
        workerRef.current.postMessage({});

        workerRef.current.onmessage = (event) => {
            const { status: workerStatus, result, error } = event.data;

            if (workerStatus === 'complete' && result) {
                displayDepthMap(result);
                // After successful completion, re-enable processing for the next frame
                setIsProcessing(false);
                setStatus('Depth map updated.');
            } else if (workerStatus === 'error') {
                console.error('Worker Error:', error);
                setStatus(`Error: ${error}. Please restart.`);
                setIsProcessing(false);
                // Stop the loop on error
                setIsCameraActive(false);
            } else {
                setStatus(workerStatus);
            }
        };

        return () => {
            if (workerRef.current) {
                workerRef.current.terminate();
            }
            // Stop camera stream on unmount
            if (videoRef.current && videoRef.current.srcObject) {
                videoRef.current.srcObject.getTracks().forEach(track => track.stop());
            }
        };
    }, []);

    // --- WEBCAM AND FRAME PROCESSING LOGIC ---

    // Function to run inference on the captured frame
    const processFrame = useCallback(() => {
        if (!isCameraActive || isProcessing || !videoRef.current.videoWidth) {
            return;
        }

        const video = videoRef.current;
        const cCanvas = captureCanvasRef.current;
        const cCtx = cCanvas.getContext('2d');

        // Set the capture canvas size to match the video feed
        cCanvas.width = video.videoWidth;
        cCanvas.height = video.videoHeight;

        // Draw the current video frame onto the hidden canvas
        cCtx.drawImage(video, 0, 0, cCanvas.width, cCanvas.height);

        // Convert the canvas content to a Blob URL (image source)
        // This is efficient as the worker can process the URL directly.
        cCanvas.toBlob((blob) => {
            if (blob) {
                const imageUrl = URL.createObjectURL(blob);

                // Signal that we are currently processing a frame
                setIsProcessing(true);
                setStatus('Running inference (WebGPU/WASM)...');

                // Send the Blob URL to the worker
                workerRef.current.postMessage({ image: imageUrl });

                // Clean up the URL object after the worker has read it
                URL.revokeObjectURL(imageUrl);
            }
        }, 'image/jpeg', 0.8); // Use JPEG for better transfer efficiency
    }, [isCameraActive, isProcessing]);


    // Controlled Processing Loop: Rerun processFrame after inference completes
    useEffect(() => {
        // If the camera is active and the worker is done processing, start the next frame capture immediately.
        if (isCameraActive && !isProcessing) {
            // Note: We use setTimeout instead of requestAnimationFrame
            // to pace the heavy ML load. 100ms is a fast minimum interval.
            const timer = setTimeout(() => {
                processFrame();
            }, 100);
            return () => clearTimeout(timer);
        }
    }, [isCameraActive, isProcessing, processFrame]);


    const startWebcam = async () => {
        if (isCameraActive) return;

        try {
            setStatus('Requesting camera access...');
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT }
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.play();
            }
            setIsCameraActive(true);
            setStatus('Camera active. Waiting for model pre-initialization...');

            // The model is pre-initialized in useEffect, once it's ready, the loop starts.

        } catch (err) {
            console.error('Error accessing webcam:', err);
            setStatus(`Camera access denied or failed: ${err.name}`);
        }
    };

    const stopWebcam = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            videoRef.current.srcObject.getTracks().forEach(track => track.stop());
        }
        setIsCameraActive(false);
        setIsProcessing(false);
        setStatus('Webcam stopped.');
    };

    // --- VISUALIZATION FUNCTION ---
    const displayDepthMap = (rawDepthData) => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // Set canvas dimensions
        canvas.width = rawDepthData.width;
        canvas.height = rawDepthData.height;

        const data = new Uint8Array(rawDepthData.data);
        const imageData = ctx.createImageData(rawDepthData.width, rawDepthData.height);

        // Iterate through the depth data and convert it to a colored image
        for (let i = 0; i < data.length; i++) {
            const pixelIndex = i * 4;
            const normalizedDepth = data[i]; // Normalized depth value (0=far, 255=near)

            const [r, g, b] = mapValueToViridisColor(normalizedDepth);

            // Set the R, G, B, and A channels
            imageData.data[pixelIndex] = r;         // R
            imageData.data[pixelIndex + 1] = g;     // G
            imageData.data[pixelIndex + 2] = b;     // B
            imageData.data[pixelIndex + 3] = 255;   // Alpha
        }

        ctx.putImageData(imageData, 0, 0);
    };

    return (
        <div className="min-h-screen bg-gray-50 p-6 font-sans">
            <script src="https://cdn.tailwindcss.com"></script>
            <div className="max-w-4xl mx-auto bg-white shadow-xl rounded-xl p-6">
                <h1 className="text-3xl font-extrabold text-indigo-700 mb-4 text-center">
                    Live Depth Anything 🚀
                </h1>
                <p className="text-center text-gray-600 mb-6">
                    Estimating depth from webcam stream using ONNX (WebGPU optimized).
                </p>

                <div className="flex justify-center space-x-4 mb-8">
                    <button
                        onClick={startWebcam}
                        disabled={isCameraActive}
                        className="px-6 py-3 bg-green-500 text-white font-bold rounded-lg shadow-md hover:bg-green-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Start Webcam
                    </button>
                    <button
                        onClick={stopWebcam}
                        disabled={!isCameraActive}
                        className="px-6 py-3 bg-red-500 text-white font-bold rounded-lg shadow-md hover:bg-red-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Stop Webcam
                    </button>
                </div>

                <div className="bg-blue-100 p-4 rounded-lg shadow-inner mb-8">
                    <p className={`font-semibold ${isProcessing ? 'text-orange-600' : 'text-gray-700'}`}>
                        Status: <span className="font-mono">{status}</span>
                    </p>
                    <p className="text-sm text-gray-500 mt-1">
                        *Note: Inference takes several seconds. The app captures a new frame only after the previous one finishes.
                    </p>
                </div>

                <div className="flex flex-col md:flex-row justify-center space-y-6 md:space-y-0 md:space-x-6">

                    {/* Live Video Feed */}
                    <div className="flex-1 min-w-0">
                        <h3 className="text-xl font-semibold mb-2 text-center">Live Input</h3>
                        <div className="relative border-4 border-gray-300 rounded-lg overflow-hidden bg-gray-800">
                            <video
                                ref={videoRef}
                                autoPlay
                                playsInline
                                muted
                                className="w-full rounded-md"
                                style={{ display: isCameraActive ? 'block' : 'none' }}
                            />
                            {!isCameraActive && (
                                <div className="absolute inset-0 flex items-center justify-center text-white text-lg">
                                    Webcam is OFF.
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Depth Map Output */}
                    <div className="flex-1 min-w-0">
                        <h3 className="text-xl font-semibold mb-2 text-center">Predicted Depth Map (Colormapped)</h3>
                        <div className="border-4 border-indigo-500 rounded-lg overflow-hidden shadow-2xl bg-gray-800">
                            <canvas
                                ref={canvasRef}
                                className="w-full h-full"
                                style={{
                                    minHeight: VIDEO_HEIGHT,
                                    display: 'block'
                                }}
                            />
                        </div>
                    </div>
                </div>

                {/* Hidden Canvas for Frame Capture */}
                <canvas ref={captureCanvasRef} style={{ display: 'none' }} />
            </div>
        </div>
    );
}

export default DepthEstimator;
