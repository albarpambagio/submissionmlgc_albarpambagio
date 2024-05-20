const tf = require('@tensorflow/tfjs-node');
const { URL } = require('url');
const path = require('path');

async function loadModel() {
    let modelUrl = process.env.MODEL_URL;

    if (!modelUrl) {
        throw new Error('MODEL_URL environment variable is not set');
    }

    // Check if MODEL_URL is a local file path and convert it to a file URL if needed
    if (modelUrl.startsWith('/') || modelUrl.startsWith('./') || modelUrl.startsWith('../')) {
        const modelPath = path.resolve(modelUrl);
        modelUrl = new URL(`file://${modelPath}`).href;
    }

    try {
        const model = await tf.loadGraphModel(modelUrl);
        console.log('Model loaded successfully');
        return model;
    } catch (error) {
        console.error('Error loading model:', error);
        throw error;
    }
}

module.exports = loadModel;
