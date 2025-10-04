// src/App.js
"use client";
import React from 'react';
import PoseEstimator from './PoseEstimator';
import DepthEstimator from './DepthEstimator';
import Depth from './Depth';

import WebcamTest from './WebcamTest'
// import './App.css';

function App() {
  return (
    <div >
      <header >

        {/* <PoseEstimator /> */}
        {/* <DepthEstimator /> */}
        <Depth />
      {/* <WebcamTest /> */}
      </header>
    </div>
  );
}

export default App;

