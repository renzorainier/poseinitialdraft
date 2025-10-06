"use client";

import React, { useEffect, useRef, useState } from "react";
import { PoseLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import * as ort from "onnxruntime-web";
import { useRouter } from "next/navigation";

// Pose skeleton connections (kept as-is)
const POSE_CONNECTIONS = [
  [11, 13], [13, 15],
  [12, 14], [14, 16],
  [11, 12], [23, 24],
  [11, 23], [12, 24],
  [23, 25], [25, 27],
  [24, 26], [26, 28],
  [27, 29], [29, 31],
  [28, 30], [30, 32],
  [7, 8], [9, 10],
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
];

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;
const MODEL_PATH = "/model.onnx";
const MODEL_KEY = "depth-model";
const MODEL_INPUT_SIZE = 518;
const CALIBRATION_FRAMES = 30;

/* ---------- IndexedDB model caching ---------- */
function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("onnx-cache", 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore("models");
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function saveModelToDB(key, bytes) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("models", "readwrite");
    tx.objectStore("models").put(bytes, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function loadModelFromDB(key) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("models", "readonly");
    const request = tx.objectStore("models").get(key);
    request.onsuccess = () => resolve(request.result || null);
    request.onerror = () => reject(request.error);
  });
}

async function loadModelCached(modelUrl, key) {
  let bytes = await loadModelFromDB(key);

  const tryEP = async (ep) => await ort.InferenceSession.create(bytes, { executionProviders: [ep] });

  if (bytes) {
    try {
      return await tryEP("webgpu");
    } catch {
      return await tryEP("wasm");
    }
  }

  const resp = await fetch(modelUrl);
  const buffer = await resp.arrayBuffer();
  bytes = new Uint8Array(buffer);
  await saveModelToDB(key, bytes);

  try {
    return await tryEP("webgpu");
  } catch {
    return await tryEP("wasm");
  }
}

/* ---------- Preprocess video frame for depth model ---------- */
function preprocessFrame(video, inputSize) {
  const canvas = document.createElement("canvas");
  canvas.width = inputSize;
  canvas.height = inputSize;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, inputSize, inputSize);
  const { data } = ctx.getImageData(0, 0, inputSize, inputSize);

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const floatData = new Float32Array(3 * inputSize * inputSize);

  for (let i = 0; i < inputSize * inputSize; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;

    floatData[i] = (r - mean[0]) / std[0];
    floatData[i + inputSize * inputSize] = (g - mean[1]) / std[1];
    floatData[i + 2 * inputSize * inputSize] = (b - mean[2]) / std[2];
  }
  return floatData;
}

/* ---------- Posture metrics (returns raw numeric metrics) ---------- */
function analyzePostureMetrics(landmarks, normData, depthW, depthH) {
  if (!landmarks) return null;

  const getDepthAt = (lm) => {
    if (!lm) return 0;
    const x = Math.floor(lm.x * depthW);
    const y = Math.floor(lm.y * depthH);
    if (x < 0 || y < 0 || x >= depthW || y >= depthH) return 0;
    return normData[y * depthW + x] ?? 0;
  };

  const HEAD = 0; // nose
  const L_SHOULDER = 11;
  const R_SHOULDER = 12;
  const L_HIP = 23;
  const R_HIP = 24;

  const head = landmarks[HEAD];
  const leftShoulder = landmarks[L_SHOULDER];
  const rightShoulder = landmarks[R_SHOULDER];
  const leftHip = landmarks[L_HIP];
  const rightHip = landmarks[R_HIP];

  if (!head || !leftShoulder || !rightShoulder || !leftHip || !rightHip) return null;

  // Depth sampling
  const headDepth = getDepthAt(head);
  const shouldersDepth = (getDepthAt(leftShoulder) + getDepthAt(rightShoulder)) / 2;
  const hipsDepth = (getDepthAt(leftHip) + getDepthAt(rightHip)) / 2;
  const torsoDepth = (shouldersDepth + hipsDepth) / 2;

  // shoulder vertical difference in pixels (relative to canvas height)
  const shoulderDiffPx = Math.abs(leftShoulder.y - rightShoulder.y) * depthH;
  // forward head metric (positive if head is closer than torso)
  const forwardHeadValue = torsoDepth - headDepth;
  // lateral head offset in pixels
  const shoulderMidX = (leftShoulder.x + rightShoulder.x) / 2;
  const headOffsetX = Math.abs(head.x - shoulderMidX) * depthW;

  return {
    shoulderDiffPx,
    forwardHeadValue,
    headOffsetX,
  };
}

/* ---------- helpers to compute mean/std ---------- */
function mean(values) {
  const n = values.length;
  if (n === 0) return 0;
  return values.reduce((s, v) => s + v, 0) / n;
}
function stddev(values, mu = null) {
  if (values.length === 0) return 0;
  const m = mu === null ? mean(values) : mu;
  const variance = values.reduce((s, v) => s + (v - m) * (v - m), 0) / values.length;
  return Math.sqrt(variance);
}

/* ---------- Main Component ---------- */
export default function PoseAndDepthCamera() {
  const videoRef = useRef(null);
  const combinedCanvasRef = useRef(null);

  const [statusMessage, setStatusMessage] = useState("Initializing...");
  const [isLoading, setIsLoading] = useState(true);
  const [postureFeedback, setPostureFeedback] = useState("");
  const [showRawCamera, setShowRawCamera] = useState(false);

  // baseline persistent (mean + std)
  const [baseline, setBaseline] = useState(() => {
    try {
      const raw = localStorage.getItem("postureBaseline");
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  });

  const [calibrating, setCalibrating] = useState(false);
  const calibratingRef = useRef(false);
  const calibrationSamplesRef = useRef([]);

  const poseLandmarkerRef = useRef(null);
  const sessionRef = useRef(null);
  const runningRef = useRef(false);

  /* ---- Draw skeleton (Head + Shoulders only) ---- */
  const drawSkeleton = (ctx, landmarks, width, height) => {
    if (!landmarks || landmarks.length === 0) return;
    const HEAD = 0;
    const L_SHOULDER = 11;
    const R_SHOULDER = 12;

    const colors = {
      [HEAD]: "#FF4500",
      [L_SHOULDER]: "#44ff1eff",
      [R_SHOULDER]: "#44ff1eff",
    };

    [HEAD, L_SHOULDER, R_SHOULDER].forEach((i) => {
      const lm = landmarks[i];
      if (!lm) return;
      const x = lm.x * width;
      const y = lm.y * height;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fillStyle = colors[i];
      ctx.fill();
    });

    const left = landmarks[L_SHOULDER];
    const right = landmarks[R_SHOULDER];
    if (left && right) {
      ctx.beginPath();
      ctx.moveTo(left.x * width, left.y * height);
      ctx.lineTo(right.x * width, right.y * height);
      ctx.strokeStyle = "#FFD700";
      ctx.lineWidth = 3;
      ctx.stroke();
    }
  };

  /* ---- Prediction loop ---- */
  const predict = async () => {
    if (!runningRef.current) return;
    const video = videoRef.current;
    const canvas = combinedCanvasRef.current;
    if (!canvas || !video) {
      requestAnimationFrame(predict);
      return;
    }
    const ctx = canvas.getContext("2d");
    // clear each frame to avoid ghosting
    //ctx.clearRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
    //ctx.drawImage(video, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);

    const poseLandmarker = poseLandmarkerRef.current;
    const session = sessionRef.current;

    let normData = null;
    let poseResults = null;

    // 1. Depth estimation
    if (session) {
      try {
        const floatData = preprocessFrame(video, MODEL_INPUT_SIZE);
        const inputTensor = new ort.Tensor("float32", floatData, [
          1,
          3,
          MODEL_INPUT_SIZE,
          MODEL_INPUT_SIZE,
        ]);

        const feeds = { [session.inputNames[0]]: inputTensor };
        const depthResults = await session.run(feeds);
        const depthOutput = depthResults[session.outputNames[0]];
        const depthData = depthOutput.data;

        // normalize depth
        let min = Infinity,
          max = -Infinity;
        for (let v of depthData) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
        // avoid division by zero
        if (max === min) {
          normData = new Array(depthData.length).fill(0);
        } else {
          normData = Array.from(depthData, (v) => (v - min) / (max - min));
        }

        // draw grayscale depth map
        const depthCanvas = document.createElement("canvas");
        depthCanvas.width = MODEL_INPUT_SIZE;
        depthCanvas.height = MODEL_INPUT_SIZE;
        const depthCtx = depthCanvas.getContext("2d");

        const imageData = depthCtx.createImageData(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
        for (let i = 0; i < normData.length; i++) {
          const val = Math.floor(normData[i] * 255);
          imageData.data[i * 4] = val;
          imageData.data[i * 4 + 1] = val;
          imageData.data[i * 4 + 2] = val;
          imageData.data[i * 4 + 3] = 150;
        }
        depthCtx.putImageData(imageData, 0, 0);
        ctx.drawImage(depthCanvas, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
      } catch (err) {
        console.error("Depth inference error:", err);
      }
    }

    // 2. Pose detection
    if (poseLandmarker) {
      try {
        poseResults = poseLandmarker.detectForVideo(video, performance.now());
        if (poseResults.landmarks && poseResults.landmarks.length > 0) {
          drawSkeleton(ctx, poseResults.landmarks[0], VIDEO_WIDTH, VIDEO_HEIGHT);
          setStatusMessage("Pose Detected");
        } else {
          setStatusMessage("No Pose Detected");
        }
      } catch (err) {
        console.error("Pose detect error:", err);
      }
    }

    // 3. Posture metrics and either calibration or live analysis
    if (poseResults?.landmarks?.length > 0 && normData) {
      const metrics = analyzePostureMetrics(
        poseResults.landmarks[0],
        normData,
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE
      );

      // 🔹 If calibrating, only collect samples (no posture feedback)
      if (calibratingRef.current && metrics) {
        calibrationSamplesRef.current.push(metrics);
        const collected = calibrationSamplesRef.current.length;
        setStatusMessage(`Calibrating... (${collected}/${CALIBRATION_FRAMES})`);

        // ⛔ Suppress feedback during calibration
        setPostureFeedback("Hold a natural, relaxed posture...");

        if (collected >= CALIBRATION_FRAMES) {
          // compute baseline
          const shoulderVals = calibrationSamplesRef.current.map((s) => s.shoulderDiffPx);
          const forwardVals = calibrationSamplesRef.current.map((s) => s.forwardHeadValue);
          const headOffsetVals = calibrationSamplesRef.current.map((s) => s.headOffsetX);

          const meanShoulder = mean(shoulderVals);
          const stdShoulder = stddev(shoulderVals, meanShoulder);
          const meanForward = mean(forwardVals);
          const stdForward = stddev(forwardVals, meanForward);
          const meanHeadOffset = mean(headOffsetVals);
          const stdHeadOffset = stddev(headOffsetVals, meanHeadOffset);

          const newBaseline = {
            meanShoulder,
            stdShoulder,
            meanForward,
            stdForward,
            meanHeadOffset,
            stdHeadOffset,
            createdAt: Date.now(),
            frames: CALIBRATION_FRAMES,
          };

          // persist and apply immediately
          try {
            localStorage.setItem("postureBaseline", JSON.stringify(newBaseline));
          } catch (e) {
            console.warn("Could not persist baseline:", e);
          }

          setBaseline(newBaseline);
          calibratingRef.current = false;
          calibrationSamplesRef.current = [];
          setCalibrating(false);
          setStatusMessage("Calibration complete ✅");
          setPostureFeedback("");
          setTimeout(() => setStatusMessage("Ready"), 1200);
        }

        // Stop here — don’t do posture checking during calibration
        requestAnimationFrame(predict);
        return;
      }

      // 🔹 Normal live posture analysis (only if not calibrating)
      if (!calibratingRef.current && metrics) {
        let shouldersNotStraight = false;
        let forwardHead = false;
        let headTilt = false;

        if (baseline) {
          const shoulderThresh = baseline.meanShoulder + Math.max(12, baseline.stdShoulder * 1.5);
          const forwardThresh = baseline.meanForward + Math.max(0.07, baseline.stdForward * 1.5);
          const headOffsetThresh = baseline.meanHeadOffset + Math.max(18, baseline.stdHeadOffset * 1.5);

          shouldersNotStraight = metrics.shoulderDiffPx > shoulderThresh;
          forwardHead = metrics.forwardHeadValue > forwardThresh;
          headTilt = metrics.headOffsetX > headOffsetThresh;
        } else {
          shouldersNotStraight = metrics.shoulderDiffPx > 20;
          forwardHead = metrics.forwardHeadValue > 0.12;
          headTilt = metrics.headOffsetX > 30;
        }

        if (shouldersNotStraight) {
          setStatusMessage("Bad Posture ❌");
          setPostureFeedback("Shoulders Uneven");
        } else if (forwardHead) {
          setStatusMessage("Bad Posture ❌");
          setPostureFeedback("Head Leaning Forward");
        } else if (headTilt) {
          setStatusMessage("Bad Posture ❌");
          setPostureFeedback("Head Tilted");
        } else {
          setStatusMessage("Good Posture ✅");
          setPostureFeedback("");
        }
      }
    }


    requestAnimationFrame(predict);
  };

  /* ---- Initialization ---- */
  useEffect(() => {
    let stream;
    const init = async () => {
      try {
        setStatusMessage("Loading models...");
        // mediapipe
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

        // load depth model
        sessionRef.current = await loadModelCached(MODEL_PATH, MODEL_KEY);

        // camera
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        runningRef.current = true;
        setIsLoading(false);
        setStatusMessage("Ready");
        requestAnimationFrame(predict);
      } catch (err) {
        console.error(err);
        setStatusMessage("Init error");
        setIsLoading(false);
      }
    };
    init();

    return () => {
      runningRef.current = false;
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ---- Calibration controls ---- */
  const startCalibration = () => {
    // require models + video
    if (!poseLandmarkerRef.current || !sessionRef.current) {
      setStatusMessage("Models not ready yet");
      return;
    }
    calibratingRef.current = true;
    calibrationSamplesRef.current = [];
    setCalibrating(true);
    setStatusMessage(`Calibrating... (0/${CALIBRATION_FRAMES})`);
    setPostureFeedback("Hold a natural 'good posture' for a few seconds");
  };

  const cancelCalibration = () => {
    calibratingRef.current = false;
    calibrationSamplesRef.current = [];
    setCalibrating(false);
    setStatusMessage("Calibration cancelled");
    setTimeout(() => setStatusMessage("Ready"), 800);
    setPostureFeedback("");
  };

  const router = useRouter();

  const goBack = () => {
    router.push("/dashboard");
  }

  const resetBaseline = () => {
    try {
      localStorage.removeItem("postureBaseline");
    } catch { }
    setBaseline(null);
    setStatusMessage("Baseline reset");
    setPostureFeedback("");
    setTimeout(() => setStatusMessage("Ready"), 800);
  };

  return (
    <div className="flex flex-col items-center justify-center p-4 min-h-screen bg-gray-900 text-white">
      <h1 className="text-3xl font-bold mb-2 text-center">Pose + Depth Estimation</h1>
      <p className="text-gray-400 mb-4 text-center">Real-time combined output</p>

      {/* Responsive camera container */}
      <div className="relative w-full max-w-[90vw] sm:max-w-[640px] aspect-[4/3] flex justify-center items-center">
        {/* Raw video */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`${showRawCamera ? "" : "hidden"} absolute top-0 left-0 w-full h-full object-cover rounded-xl bg-black`}
        />

        {/* Processed canvas */}
        <canvas
          ref={combinedCanvasRef}
          width={VIDEO_WIDTH}
          height={VIDEO_HEIGHT}
          className={`${!showRawCamera ? "" : "hidden"} absolute top-0 left-0 w-full h-full object-cover rounded-xl bg-black`}
        />
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-3 justify-center mt-6">
        <button
          onClick={() => setShowRawCamera((s) => !s)}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold"
        >
          {showRawCamera ? "Show Processed View" : "Show Raw Camera"}
        </button>

        {!calibrating ? (
          <button
            onClick={startCalibration}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg font-semibold"
            title="Stand in your natural good posture and click Calibrate"
          >
            Calibrate
          </button>
        ) : (
          <button
            onClick={cancelCalibration}
            className="px-4 py-2 bg-orange-500 hover:bg-orange-600 rounded-lg font-semibold"
          >
            Cancel Calibration
          </button>
        )}

        <button
          onClick={resetBaseline}
          className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg font-semibold"
        >
          Reset Baseline
        </button>
        <button
          onClick={goBack}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-800 rounded-lg font-semibold"
        >
          Dashboard
        </button>
      </div>

      {/* Status / baseline info */}
      <div
        className={`mt-6 p-3 rounded-lg font-bold text-center ${statusMessage.includes("Detected") || statusMessage.includes("Good")
          ? "bg-green-600"
          : "bg-yellow-600"
          }`}
        style={{ minWidth: 280 }}
      >
        {isLoading ? "Loading..." : statusMessage}
        {postureFeedback && <p className="mt-2 text-red-400">{postureFeedback}</p>}
        {baseline && (
          <div className="mt-2 text-sm text-gray-200 text-left">
            <div><strong>Baseline:</strong></div>
            <div>Shoulder mean: {baseline.meanShoulder.toFixed(1)} px (σ {baseline.stdShoulder.toFixed(2)})</div>
            <div>Forward mean: {baseline.meanForward.toFixed(3)} (σ {baseline.stdForward.toFixed(3)})</div>
            <div>Head offset mean: {baseline.meanHeadOffset.toFixed(1)} px (σ {baseline.stdHeadOffset.toFixed(2)})</div>
          </div>
        )}
      </div>
    </div>
  );

}
