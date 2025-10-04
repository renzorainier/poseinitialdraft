"use client";

import React, { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

// --- IndexedDB helpers ---
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

// preprocess helper for raw video frame
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

// model loader with caching
async function loadModelCached(modelUrl, key) {
  let bytes = await loadModelFromDB(key);
  const tryEP = async (ep) =>
    await ort.InferenceSession.create(bytes, { executionProviders: [ep] });

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

export default function DepthWebcam() {
  const [session, setSession] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const MODEL_PATH = "/model.onnx";
  const MODEL_KEY = "depth-model";
  const MODEL_INPUT_SIZE = 518;

  const TARGET_FPS = 1; // adjust this if needed
  const FRAME_INTERVAL = 1000 / TARGET_FPS;

  // load model once
  useEffect(() => {
    async function init() {
      const newSession = await loadModelCached(MODEL_PATH, MODEL_KEY);
      setSession(newSession);
    }
    init();
  }, []);

  // init webcam
  useEffect(() => {
    async function initCamera() {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    }
    initCamera();
  }, []);

  // run inference loop with throttled FPS
  useEffect(() => {
    if (!session || !videoRef.current || !canvasRef.current) return;

    let stop = false;
    let lastTime = 0;

    async function loop(timestamp) {
      if (stop) return;
      if (timestamp - lastTime >= FRAME_INTERVAL) {
        lastTime = timestamp;

        const floatData = preprocessFrame(videoRef.current, MODEL_INPUT_SIZE);
        const inputTensor = new ort.Tensor("float32", floatData, [
          1,
          3,
          MODEL_INPUT_SIZE,
          MODEL_INPUT_SIZE,
        ]);
        const feeds = {};
        feeds[session.inputNames[0]] = inputTensor;

        try {
          const results = await session.run(feeds);
          const output = results[session.outputNames[0]];
          const depthData = output.data;

          // normalize
          let min = Infinity,
            max = -Infinity;
          for (let v of depthData) {
            if (v < min) min = v;
            if (v > max) max = v;
          }
          const normData = depthData.map((v) => (v - min) / (max - min));

          // render depth
          const ctx = canvasRef.current.getContext("2d");
          const imageData = ctx.createImageData(
            MODEL_INPUT_SIZE,
            MODEL_INPUT_SIZE
          );
          for (let i = 0; i < normData.length; i++) {
            const val = Math.floor(normData[i] * 255);
            imageData.data[i * 4] = val;
            imageData.data[i * 4 + 1] = val;
            imageData.data[i * 4 + 2] = val;
            imageData.data[i * 4 + 3] = 255;
          }
          ctx.putImageData(imageData, 0, 0);
        } catch (err) {
          console.error("inference error:", err);
        }
      }

      requestAnimationFrame(loop);
    }

    requestAnimationFrame(loop);
    return () => {
      stop = true;
    };
  }, [session]);

  return (
    <div className="flex flex-col items-center p-4">
      <h1 className="text-xl font-bold mb-4">Webcam Depth Estimation</h1>

      <div className="grid grid-cols-2 gap-4">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="rounded-xl w-[518px] h-[518px] bg-black"
        />
        <canvas
          ref={canvasRef}
          width={MODEL_INPUT_SIZE}
          height={MODEL_INPUT_SIZE}
          className="rounded-xl"
        />
      </div>
    </div>
  );
}

// "use client";


// import React, { useEffect, useRef, useState } from "react";
// import * as ort from "onnxruntime-web";

// // --- IndexedDB helpers ---
// function openDB() {
//   return new Promise((resolve, reject) => {
//     const request = indexedDB.open("onnx-cache", 1);
//     request.onupgradeneeded = () => {
//       request.result.createObjectStore("models");
//     };
//     request.onsuccess = () => resolve(request.result);
//     request.onerror = () => reject(request.error);
//   });
// }

// async function saveModelToDB(key, bytes) {
//   const db = await openDB();
//   return new Promise((resolve, reject) => {
//     const tx = db.transaction("models", "readwrite");
//     tx.objectStore("models").put(bytes, key);
//     tx.oncomplete = () => resolve();
//     tx.onerror = () => reject(tx.error);
//   });
// }

// async function loadModelFromDB(key) {
//   const db = await openDB();
//   return new Promise((resolve, reject) => {
//     const tx = db.transaction("models", "readonly");
//     const request = tx.objectStore("models").get(key);
//     request.onsuccess = () => resolve(request.result || null);
//     request.onerror = () => reject(request.error);
//   });
// }

// // preprocess helper for raw video frame
// function preprocessFrame(video, inputSize) {
//   const canvas = document.createElement("canvas");
//   canvas.width = inputSize;
//   canvas.height = inputSize;
//   const ctx = canvas.getContext("2d");
//   ctx.drawImage(video, 0, 0, inputSize, inputSize);
//   const { data } = ctx.getImageData(0, 0, inputSize, inputSize);

//   const mean = [0.485, 0.456, 0.406];
//   const std = [0.229, 0.224, 0.225];
//   const floatData = new Float32Array(3 * inputSize * inputSize);

//   for (let i = 0; i < inputSize * inputSize; i++) {
//     const r = data[i * 4] / 255;
//     const g = data[i * 4 + 1] / 255;
//     const b = data[i * 4 + 2] / 255;

//     floatData[i] = (r - mean[0]) / std[0];
//     floatData[i + inputSize * inputSize] = (g - mean[1]) / std[1];
//     floatData[i + 2 * inputSize * inputSize] = (b - mean[2]) / std[2];
//   }

//   return floatData;
// }

// // model loader
// async function loadModelCached(modelUrl, key) {
//   let bytes = await loadModelFromDB(key);
//   const tryEP = async (ep) =>
//     await ort.InferenceSession.create(bytes, { executionProviders: [ep] });

//   if (bytes) {
//     try {
//       return await tryEP("webgpu");
//     } catch {
//       return await tryEP("wasm");
//     }
//   }

//   const resp = await fetch(modelUrl);
//   const buffer = await resp.arrayBuffer();
//   bytes = new Uint8Array(buffer);
//   await saveModelToDB(key, bytes);

//   try {
//     return await tryEP("webgpu");
//   } catch {
//     return await tryEP("wasm");
//   }
// }

// export default function DepthWebcam() {
//   const [session, setSession] = useState(null);
//   const videoRef = useRef(null);
//   const canvasRef = useRef(null);

//   const MODEL_PATH = "/model.onnx";
//   const MODEL_KEY = "depth-model";
//   const MODEL_INPUT_SIZE = 518;

//   // load model once
//   useEffect(() => {
//     async function init() {
//       const newSession = await loadModelCached(MODEL_PATH, MODEL_KEY);
//       setSession(newSession);
//     }
//     init();
//   }, []);

//   // init webcam
//   useEffect(() => {
//     async function initCamera() {
//       const stream = await navigator.mediaDevices.getUserMedia({ video: true });
//       if (videoRef.current) {
//         videoRef.current.srcObject = stream;
//         await videoRef.current.play();
//       }
//     }
//     initCamera();
//   }, []);

//   // run inference on webcam frames
//   useEffect(() => {
//     if (!session || !videoRef.current || !canvasRef.current) return;

//     let stop = false;

//     async function loop() {
//       if (stop) return;

//       const floatData = preprocessFrame(videoRef.current, MODEL_INPUT_SIZE);
//       const inputTensor = new ort.Tensor("float32", floatData, [
//         1,
//         3,
//         MODEL_INPUT_SIZE,
//         MODEL_INPUT_SIZE,
//       ]);
//       const feeds = {};
//       feeds[session.inputNames[0]] = inputTensor;

//       try {
//         const results = await session.run(feeds);
//         const output = results[session.outputNames[0]];
//         const depthData = output.data;

//         // normalize
//         let min = Infinity,
//           max = -Infinity;
//         for (let v of depthData) {
//           if (v < min) min = v;
//           if (v > max) max = v;
//         }
//         const normData = depthData.map((v) => (v - min) / (max - min));

//         // render depth
//         const ctx = canvasRef.current.getContext("2d");
//         const imageData = ctx.createImageData(
//           MODEL_INPUT_SIZE,
//           MODEL_INPUT_SIZE
//         );
//         for (let i = 0; i < normData.length; i++) {
//           const val = Math.floor(normData[i] * 255);
//           imageData.data[i * 4] = val;
//           imageData.data[i * 4 + 1] = val;
//           imageData.data[i * 4 + 2] = val;
//           imageData.data[i * 4 + 3] = 255;
//         }
//         ctx.putImageData(imageData, 0, 0);
//       } catch (err) {
//         console.error("inference error:", err);
//       }

//       requestAnimationFrame(loop);
//     }

//     loop();
//     return () => {
//       stop = true;
//     };
//   }, [session]);

//   return (
//     <div className="flex flex-col items-center p-4">
//       <h1 className="text-xl font-bold mb-4">Webcam Depth Estimation</h1>

//       <div className="grid grid-cols-2 gap-4">
//         <video
//           ref={videoRef}
//           autoPlay
//           playsInline
//           muted
//           className="rounded-xl w-[518px] h-[518px] bg-black"
//         />
//         <canvas
//           ref={canvasRef}
//           width={MODEL_INPUT_SIZE}
//           height={MODEL_INPUT_SIZE}
//           className="rounded-xl"
//         />
//       </div>
//     </div>
//   );
// }



// "use client";



// import React, { useEffect, useState } from "react";
// import * as ort from "onnxruntime-web";

// // --- IndexedDB helpers ---
// function openDB() {
//   return new Promise((resolve, reject) => {
//     const request = indexedDB.open("onnx-cache", 1);
//     request.onupgradeneeded = () => {
//       request.result.createObjectStore("models");
//     };
//     request.onsuccess = () => resolve(request.result);
//     request.onerror = () => reject(request.error);
//   });
// }

// async function saveModelToDB(key, bytes) {
//   const db = await openDB();
//   return new Promise((resolve, reject) => {
//     const tx = db.transaction("models", "readwrite");
//     tx.objectStore("models").put(bytes, key);
//     tx.oncomplete = () => resolve();
//     tx.onerror = () => reject(tx.error);
//   });
// }

// async function loadModelFromDB(key) {
//   const db = await openDB();
//   return new Promise((resolve, reject) => {
//     const tx = db.transaction("models", "readonly");
//     const request = tx.objectStore("models").get(key);
//     request.onsuccess = () => resolve(request.result || null);
//     request.onerror = () => reject(request.error);
//   });
// }

// // --- preprocess helper ---
// async function preprocessImage(fileOrUrl, inputSize) {
//   return new Promise((resolve) => {
//     const img = new Image();
//     img.crossOrigin = "anonymous";

//     if (fileOrUrl instanceof File) {
//       img.src = URL.createObjectURL(fileOrUrl);
//     } else {
//       img.src = fileOrUrl;
//     }

//     img.onload = () => {
//       const canvas = document.createElement("canvas");
//       canvas.width = inputSize;
//       canvas.height = inputSize;
//       const ctx = canvas.getContext("2d");
//       ctx.drawImage(img, 0, 0, inputSize, inputSize);
//       const { data } = ctx.getImageData(0, 0, inputSize, inputSize);

//       const mean = [0.485, 0.456, 0.406];
//       const std = [0.229, 0.224, 0.225];
//       const floatData = new Float32Array(3 * inputSize * inputSize);

//       for (let i = 0; i < inputSize * inputSize; i++) {
//         const r = data[i * 4] / 255;
//         const g = data[i * 4 + 1] / 255;
//         const b = data[i * 4 + 2] / 255;

//         floatData[i] = (r - mean[0]) / std[0];
//         floatData[i + inputSize * inputSize] = (g - mean[1]) / std[1];
//         floatData[i + 2 * inputSize * inputSize] = (b - mean[2]) / std[2];
//       }

//       resolve({ floatData, originalUrl: img.src });
//     };
//   });
// }

// // --- model loader (IndexedDB cache) ---
// async function loadModelCached(modelUrl, key) {
//   let bytes = await loadModelFromDB(key);
//   const tryEP = async (ep) => {
//     return await ort.InferenceSession.create(bytes, { executionProviders: [ep] });
//   };

//   if (bytes) {
//     console.log("loaded model from IndexedDB cache");
//     try {
//       console.log("trying WebGPU EP...");
//       return await tryEP("webgpu");
//     } catch (err) {
//       console.warn("GPU init failed, falling back to WASM:", err);
//       return await tryEP("wasm");
//     }
//   }

//   console.log("fetching model from network...");
//   const resp = await fetch(modelUrl);
//   const buffer = await resp.arrayBuffer();
//   bytes = new Uint8Array(buffer);
//   await saveModelToDB(key, bytes);
//   console.log("model saved in IndexedDB");

//   try {
//     console.log("trying WebGPU EP...");
//     return await tryEP("webgpu");
//   } catch (err) {
//     console.warn("GPU init failed, falling back to WASM:", err);
//     return await tryEP("wasm");
//   }
// }

// export default function Depth() {
//   const [session, setSession] = useState(null);
//   const [outputUrl, setOutputUrl] = useState(null);
//   const [originalUrl, setOriginalUrl] = useState("/test_image.jpg");
//   const [loading, setLoading] = useState(false);

//   const MODEL_PATH = "/model.onnx"; // model placed in public/
//   const MODEL_KEY = "depth-model";
//   const MODEL_INPUT_SIZE = 518;

//   // load model once
//   useEffect(() => {
//     async function init() {
//       try {
//         const newSession = await loadModelCached(MODEL_PATH, MODEL_KEY);
//         setSession(newSession);
//         console.log("model ready");
//       } catch (err) {
//         console.error("failed to load model:", err);
//       }
//     }
//     init();
//   }, []);

//   async function runInference(imageFileOrUrl) {
//     if (!session) return;
//     setLoading(true);

//     try {
//       const { floatData, originalUrl } = await preprocessImage(
//         imageFileOrUrl,
//         MODEL_INPUT_SIZE
//       );
//       setOriginalUrl(originalUrl);

//       const inputTensor = new ort.Tensor("float32", floatData, [
//         1,
//         3,
//         MODEL_INPUT_SIZE,
//         MODEL_INPUT_SIZE,
//       ]);

//       const feeds = {};
//       feeds[session.inputNames[0]] = inputTensor;

//       const results = await session.run(feeds);
//       const output = results[session.outputNames[0]];
//       const depthData = output.data;

//       let min = Infinity,
//         max = -Infinity;
//       for (let v of depthData) {
//         if (v < min) min = v;
//         if (v > max) max = v;
//       }
//       const normData = depthData.map((v) => (v - min) / (max - min));

//       const canvas = document.createElement("canvas");
//       canvas.width = MODEL_INPUT_SIZE;
//       canvas.height = MODEL_INPUT_SIZE;
//       const ctx = canvas.getContext("2d");
//       const imageData = ctx.createImageData(
//         MODEL_INPUT_SIZE,
//         MODEL_INPUT_SIZE
//       );
//       for (let i = 0; i < normData.length; i++) {
//         const val = Math.floor(normData[i] * 255);
//         imageData.data[i * 4] = val;
//         imageData.data[i * 4 + 1] = val;
//         imageData.data[i * 4 + 2] = val;
//         imageData.data[i * 4 + 3] = 255;
//       }
//       ctx.putImageData(imageData, 0, 0);
//       setOutputUrl(canvas.toDataURL());
//     } catch (err) {
//       console.error("inference failed:", err);
//     } finally {
//       setLoading(false);
//     }
//   }

//   // run once with default image
//   useEffect(() => {
//     if (session) runInference("/test_image.jpg");
//   }, [session]);

//   return (
//     <div className="flex flex-col items-center p-4">
//       <h1 className="text-xl font-bold mb-4">ONNX Depth Estimation</h1>

//       <input
//         type="file"
//         accept="image/*"
//         className="mb-4"
//         onChange={(e) => {
//           if (e.target.files && e.target.files[0]) {
//             runInference(e.target.files[0]);
//           }
//         }}
//       />

//       {loading && <p>running inference...</p>}

//       {!loading && outputUrl && (
//         <div className="grid grid-cols-2 gap-4">
//           <div>
//             <h2 className="mb-2 font-semibold">Original</h2>
//             <img src={originalUrl} alt="original" className="rounded-xl" />
//           </div>
//           <div>
//             <h2 className="mb-2 font-semibold">Predicted Depth</h2>
//             <img src={outputUrl} alt="depth" className="rounded-xl" />
//           </div>
//         </div>
//       )}
//     </div>
//   );
// }
