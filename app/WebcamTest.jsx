// src/components/WebcamTest.jsx
"use client";
import React, { useEffect, useRef } from "react";

export default function WebcamTest() {
  const videoRef = useRef(null);

  useEffect(() => {
    // ask for webcam
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        console.log("✅ Got webcam stream:", stream);
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            console.log("📸 Video metadata loaded:",
              videoRef.current.videoWidth,
              videoRef.current.videoHeight
            );
            videoRef.current.play();
          };
        }
      })
      .catch((err) => {
        console.error("❌ Error accessing webcam:", err);
      });
  }, []);

  return (
    <div>
      <h2>Webcam Test</h2>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          width: "640px",
          height: "480px",
          background: "black", // makes it obvious if no video
        }}
      />
    </div>
  );
}
