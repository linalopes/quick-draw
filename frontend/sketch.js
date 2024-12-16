let canvas;

function setup() {
    const canvasContainer = document.getElementById('canvas-container');
    canvas = createCanvas(280, 280);  // Drawing area
    canvas.parent(canvasContainer);
    background(255);  // White background
}

function draw() {
    strokeWeight(10);  // Pen thickness
    stroke(0);  // Black color
    if (mouseIsPressed) {
        line(mouseX, mouseY, pmouseX, pmouseY);
    }
}

function clearCanvas() {
    background(255);  // Clear the canvas
}

function sendDrawing() {
    canvas.elt.toBlob((blob) => {
        const formData = new FormData();
        formData.append('image', blob, 'sketch.png');

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('result').innerText = 
                data.map(pred => `${pred.class}: ${pred.probability}%`).join("\n");
        })
        .catch(error => console.error('Error:', error));
    });
}

function preprocessAndShow() {
    canvas.elt.toBlob((blob) => {
        const formData = new FormData();
        formData.append('image', blob, 'sketch.png');

        fetch('http://127.0.0.1:5000/preprocess', {
            method: 'POST',
            body: formData
        })
        .then(response => response.blob())
        .then(blob => {
            const url = URL.createObjectURL(blob);
            document.getElementById('processedImage').src = url;  // Display the image
        })
        .catch(error => console.error('Error:', error));
    });
}