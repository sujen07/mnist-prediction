const CANVAS_SIZE = 280;
const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
let isMouseDown = false;
let hasIntroText = true;
let lastX = 0;
let lastY = 0;
ctx.lineWidth = 25;
ctx.strokeStyle = "#333"; /* Darker stroke color for better contrast */
ctx.lineJoin = "round";

async function loadModel() {
    const sess = await ort.InferenceSession.create('./model.onnx');
    return sess;
}

function drawLine(fromX, fromY, toX, toY) {
    if (hasIntroText) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        hasIntroText = false;
    }
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.closePath();
    ctx.stroke();
    updatePredictions();
}

function clearCanvas() {
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    hasIntroText = true;
    resetRadialProgressBars();
}

function canvasMouseDown(event) {
    isMouseDown = true;
    const x = event.offsetX;
    const y = event.offsetY;
    lastX = x + 0.001;
    lastY = y + 0.001;
    canvasMouseMove(event);
}

function canvasMouseMove(event) {
    const x = event.offsetX;
    const y = event.offsetY;
    if (isMouseDown) {
        drawLine(lastX, lastY, x, y);
    }
    lastX = x;
    lastY = y;
}

function bodyMouseUp() {
    isMouseDown = false;
    updatePredictions();
}

function bodyMouseOut(event) {
    if (!event.relatedTarget || event.relatedTarget.nodeName === "HTML") {
        isMouseDown = false;
    }
}

let session;
loadModel().then(sess => {
    session = sess;
    canvas.addEventListener("mousedown", canvasMouseDown);
    canvas.addEventListener("mousemove", canvasMouseMove);
    document.body.addEventListener("mouseup", bodyMouseUp);
    document.body.addEventListener("mouseout", bodyMouseOut);
    document.getElementById("clearButton").addEventListener("click", clearCanvas);
});

function updateRadialProgressBars(predictions) {
    for (let i = 0; i < 10; i++) {
        const progressBar = document.getElementById(`digit-${i}`);
        const progressValue = Math.round(predictions[i] * 100);
        progressBar.style.setProperty('--progress', progressValue);
        progressBar.innerHTML = `
            <div class="digit-label">${i}</div>
            <div class="prediction-value">${progressValue}%</div>
        `;
    }
}

function resetRadialProgressBars() {
    for (let i = 0; i < 10; i++) {
        const progressBar = document.getElementById(`digit-${i}`);
        progressBar.style.setProperty('--progress', 0);
        progressBar.innerHTML = `
            <div class="digit-label">${i}</div>
            <div class="prediction-value">0%</div>
        `;
    }
}

async function updatePredictions() {
    const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const inputTensor = new ort.Tensor(new Float32Array(imgData.data));
    const outputMap = await session.run({ input: inputTensor });
    const outputKey = Object.keys(outputMap)[0];
    const predictions = outputMap[outputKey].cpuData;
    updateRadialProgressBars(predictions);
}
