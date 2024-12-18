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
    toggleImageVisibility(false);
    document.getElementById('donut-chart').innerHTML = '';  // Clear the donut chart
    document.getElementById('result').innerText = '';  // Clear the results
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
            const top3 = data.slice(0, 3);
            document.getElementById('result').innerText = 
                top3.map(pred => `${pred.class}: ${pred.probability}%`).join("\n");

            drawDonutChart(top3);
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
            toggleImageVisibility(true);  // Ensure the image container is visible
        })
        .catch(error => console.error('Error:', error));
    });
}

function toggleImageVisibility(show) {
    const imageContainer = document.getElementById('processedImageContainer');
    if (show) {
        imageContainer.classList.remove('d-none');
    } else {
        imageContainer.classList.add('d-none');
    }
}

function saveDrawing() {
    canvas.elt.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'sketch.png';
        a.click();
    });
}

function drawDonutChart(data) {
    const width = 350;
    const height = 350;
    const margin = 20;

    const radius = Math.min(width, height) / 2 - margin;

    const svg = d3.select("#donut-chart")
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", `translate(${width / 2},${height / 2})`);

    const color = d3.scaleOrdinal()
        .domain(data.map(d => d.class))
        .range(['deep-purple', 'pink', 'turquoise']);

    const pie = d3.pie()
        .value(d => d.probability);

    const data_ready = pie(data);

    const arc = d3.arc()
        .innerRadius(radius * 0.5)
        .outerRadius(radius * 0.8);

    const outerArc = d3.arc()
        .innerRadius(radius * 0.9)
        .outerRadius(radius * 0.9);

    svg
        .selectAll('whatever')
        .data(data_ready)
        .join('path')
        .attr('d', arc)
        .attr('class', d => color(d.data.class))
        .attr("stroke", "white")
        .style("stroke-width", "2px")
        .style("opacity", 0.7);

    svg
        .selectAll('whatever')
        .data(data_ready)
        .join('text')
        .text(d => d.data.class)
        .attr("transform", d => `translate(${outerArc.centroid(d)})`)
        .style("text-anchor", "middle")
        .style("font-size", 17);
}