let session;

async function loadModel() {
    // Make sure to adjust the path to where your model is stored in the repository
    const response = await fetch('slidevqa_best.onnx');
    const arrayBuffer = await response.arrayBuffer();
    session = await ort.InferenceSession.create(arrayBuffer);
}

document.getElementById('image-input').onchange = async function (event) {
    const file = event.target.files[0];
    const imgElement = document.getElementById('uploaded-image');
    imgElement.src = URL.createObjectURL(file);
    imgElement.onload = function() {
        document.getElementById('run-inference').style.display = 'inline';
    };
    imgElement.style.display = 'block';
};

document.getElementById('run-inference').onclick = async function () {
    const imgElement = document.getElementById('uploaded-image');
    const resizedTensor = await ort.Tensor.fromImage(imgElement, options = {resizedWidth: 640, resizedHeight: 640});
    const tensor = imageToTensor(imgElement); // Convert image to tensor
    const feeds = { input: resizedTensor };
    const results = await session.run(feeds);
    const outputTensor = results.output;
    visualizeResults(imgElement, outputTensor); // Visualize the results
};

async function imageToTensor(imgElement) {
    // const htmlTensor = await ort.Tensor.fromImage(imgElement);
    const resizedTensor = await ort.Tensor.fromImage(imgElement, options = {resizedWidth: 640, resizedHeight: 640});
    return resizedTensor;
}

function visualizeResults(imgElement, outputTensor) {
    // Implementation needed to visualize the results
}

loadModel(); // Load the model as soon as the script runs
