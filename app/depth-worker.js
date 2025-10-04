import { pipeline } from '@huggingface/transformers';

// The ONNX version of Depth Anything Small
const MODEL_NAME = 'Xenova/depth-anything-small-hf';

/**
 * Singleton class to ensure the model is loaded only once.
 */
class DepthEstimationPipeline {
    static instance = null;

    static async getInstance() {
        if (this.instance === null) {
            // Initialize the pipeline for depth estimation.
            // Explicitly set the device to 'webgpu' for massive speedup.
            // The library will fall back to CPU/WASM if WebGPU is unavailable.
            self.postMessage({ status: 'Initializing model...' });
            this.instance = await pipeline('depth-estimation', MODEL_NAME, {
                device: 'webgpu'
            });
            self.postMessage({ status: 'Model initialized. Ready for video stream.' });
        }
        return this.instance;
    }
}

// Event listener to receive messages from the main React thread
self.addEventListener('message', async (event) => {
    const { image } = event.data;

    if (!image) {
        // This is likely the initialization call.
        await DepthEstimationPipeline.getInstance();
        return;
    }

    self.postMessage({ status: 'running inference' });

    try {
        const estimator = await DepthEstimationPipeline.getInstance();

        // Run the inference. 'image' is a Blob URL created from the video frame.
        const output = await estimator(image);

        // Extract transferable data from the result
        const depthImage = output.depth;

        // The data is a Uint8Array, we convert it to an ArrayBuffer
        // for efficient transfer back to the main thread.
        const transferableResult = {
            data: depthImage.data.buffer,
            width: depthImage.width,
            height: depthImage.height,
        };

        // Send the simple object and the ArrayBuffer back using the 'transfer list'
        self.postMessage({ status: 'complete', result: transferableResult }, [transferableResult.data]);

    } catch (error) {
        // Send a simple error message string back
        console.error("Worker failed:", error);
        self.postMessage({ status: 'error', error: error.message });
    }
});
