"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  PoseLandmarker,
  FilesetResolver,
} from "@mediapipe/tasks-vision";

// Define the standard MediaPipe Pose connections for drawing the skeleton
const POSE_CONNECTIONS = [
  [11, 13], [13, 15], // Left arm (Shoulder, Elbow, Wrist)
  [12, 14], [14, 16], // Right arm
  [11, 12],           // Shoulders
  [23, 24],           // Hips
  [11, 23], [12, 24], // Torso
  [23, 25], [25, 27], // Left leg (Hip, Knee, Ankle)
  [24, 26], [26, 28], // Right leg
  [27, 29], [29, 31], // Left foot
  [28, 30], [30, 32], // Right foot
  [7, 8],             // Ears
  [9, 10],            // Eyes
  [0, 1], [1, 2], [2, 3], [3, 7], // Left face
  [0, 4], [4, 5], [5, 6], [6, 8], // Right face
];

// Fixed dimensions for the video and canvas elements
const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;

export default function PoseEstimator2D() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null); // Ref for the overlay canvas

  const [isLoading, setIsLoading] = useState(true);
  const [statusMessage, setStatusMessage] = useState("Initializing..."); // Status replaces 'postureStatus'

  const poseLandmarkerRef = useRef(null);
  const runningRef = useRef(false);

  // --- Utility Drawing Functions ---

  /**
   * Draws the skeleton connections and landmarks onto the canvas.
   */
  const drawLandmarks = (ctx, landmarks, width, height) => {
    ctx.clearRect(0, 0, width, height);

    if (!landmarks || landmarks.length === 0) return;

    // 1. Draw connections (skeleton)
    ctx.lineWidth = 4;
    // Set a fixed color for the skeleton (Cornflower Blue)
    ctx.strokeStyle = 'rgba(100, 149, 237, 0.8)';

    POSE_CONNECTIONS.forEach(([i, j]) => {
      const p1 = landmarks[i];
      const p2 = landmarks[j];

      if (p1 && p2) {
        ctx.beginPath();
        // Use p.x directly for non-mirrored view
        ctx.moveTo(p1.x * width, p1.y * height);
        ctx.lineTo(p2.x * width, p2.y * height);
        ctx.stroke();
      }
    });

    // 2. Draw landmarks (points)
    const landmarkColor = '#FFD700'; // Gold color for landmarks

    landmarks.forEach((lm) => {
      // Use lm.x directly for non-mirrored view
      const x = lm.x * width;
      const y = lm.y * height;

      ctx.fillStyle = landmarkColor;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI); // Radius of 6px
      ctx.fill();
    });
  };

  // --- Prediction Loop ---

  const predictWebcam = () => {
    if (!runningRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const poseLandmarker = poseLandmarkerRef.current;
    const ctx = canvas.getContext('2d');

    if (!video || !poseLandmarker || !ctx) {
      requestAnimationFrame(predictWebcam);
      return;
    }

    // Perform detection
    const results = poseLandmarker.detectForVideo(video, performance.now());

    if (results.landmarks && results.landmarks.length > 0) {
      const landmarks = results.landmarks[0];
      // Draw skeleton
      drawLandmarks(ctx, landmarks, VIDEO_WIDTH, VIDEO_HEIGHT);
      // Update status
      setStatusMessage("Pose Detected");
    } else {
      // Clear canvas and update status if no pose is found
      drawLandmarks(ctx, [], VIDEO_WIDTH, VIDEO_HEIGHT);
      setStatusMessage("No Pose Detected. Ensure full body is visible.");
    }

    requestAnimationFrame(predictWebcam);
  };

  // --- Initialization Effect ---

  useEffect(() => {
    let stream;

    const init = async () => {
      try {
        setStatusMessage("Loading model...");
        // 1. Initialize MediaPipe
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm"
        );
        poseLandmarkerRef.current = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numPoses: 1,
        });

        setStatusMessage("Starting camera...");
        // 2. Start Webcam
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT }
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
            runningRef.current = true;
            setStatusMessage("Ready. Detecting pose...");
            requestAnimationFrame(predictWebcam);
          };
        }

        setIsLoading(false);
      } catch (err) {
        console.error("Init error:", err);
        setStatusMessage("Error: Could not load model or access camera.");
        setIsLoading(false);
      }
    };

    init();

    return () => {
      runningRef.current = false;
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // --- JSX Rendering ---

  return (
    <div className="flex flex-col items-center justify-center p-4 min-h-screen bg-gray-900">
        <script src="https://cdn.tailwindcss.com"></script>
        <style dangerouslySetInnerHTML={{__html: `
            .video-container {
                position: relative;
                width: ${VIDEO_WIDTH}px;
                height: ${VIDEO_HEIGHT}px;
                border-radius: 1rem;
                overflow: hidden;
                box-shadow: 0 10px 40px rgba(100, 149, 237, 0.4);
                background-color: #000;
            }
            .video-feed, .pose-canvas {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                /* Non-mirrored view */
            }
            /* Make the container responsive for smaller screens */
            @media (max-width: 768px) {
                .video-container {
                    width: 95vw;
                    height: calc(95vw * ${VIDEO_HEIGHT / VIDEO_WIDTH});
                }
            }
        `}} />
        <h1 className="text-4xl font-extrabold mb-2 text-white">2D Pose Detector</h1>
        <p className="mb-6 text-gray-400">Real-time skeleton tracking using MediaPipe.</p>

        {/* Webcam and Canvas Overlay Container */}
        <div className="video-container">
            {/* The video element provides the raw image feed */}
            <video ref={videoRef} className="video-feed" playsInline muted />
            {/* The canvas element sits on top of the video for drawing */}
            <canvas ref={canvasRef} className="pose-canvas" width={VIDEO_WIDTH} height={VIDEO_HEIGHT} />
        </div>

        {/* Status Display Card */}
        <div className={`mt-8 p-4 rounded-xl shadow-xl font-bold text-lg w-full max-w-lg text-center
            ${isLoading ? 'bg-gray-700 text-gray-300' :
             statusMessage.includes("Detected") ? 'bg-green-600 text-white' :
             'bg-yellow-600 text-white'}`}>
            {statusMessage}
        </div>
    </div>
  );
}
