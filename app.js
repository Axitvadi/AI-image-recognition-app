const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

const app = express();

// Load the trained model
let model;
const classLabels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"];

// Configure Multer for file upload
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.use(express.static('public')); // Assuming your 'index.html' resides in a 'public' directory

app.get('/', (req, res) => {
    res.sendFile(__dirname + '/public/index.html');
});

app.post('/predict', upload.single('image'), async (req, res) => {
    try {
        const imageBuffer = req.file.buffer;
        const tensorImage = tf.node.decodeImage(new Uint8Array(imageBuffer), 3);
        const grayImage = tensorImage.mean(2);
        const resizedImage = tf.image.resizeBilinear(grayImage, [28, 28]);
        const normalizedImage = resizedImage.div(255.0);
        const imageBatch = normalizedImage.expandDims(0);
        
        const prediction = model.predict(imageBatch);
        const predictedClass = prediction.argMax(-1).dataSync()[0];
        const predictedLabel = classLabels[predictedClass];
        
        res.json({ prediction: predictedLabel });
    } catch (e) {
        res.json({ error: e.message });
    }
});

async function loadModel() {
    model = await tf.loadLayersModel('file://./finalproject.json'); // Assuming model is saved in TensorFlow.js format
}

loadModel().then(() => {
    app.listen(4000, () => {
        console.log('Server started on http://localhost:3000');
    });
});
