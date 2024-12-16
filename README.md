# QuickDraw Exercise
This repository is a reimplementation of the QuickDraw project as part of an assignment for my CAS - AI for Creative Practices. It focuses on training, evaluating, and deploying a simple neural network for creative applications.

[Google Colab link](https://colab.research.google.com/github/linalopes/quick-draw/blob/main/Quick-Main.ipynb)

## What is QuickDraw?
QuickDraw is a large-scale dataset of doodles collected from the Quick, Draw! game by Google, where players are tasked with quickly sketching objects. You can find the original project and dataset here:
[QuickDraw Dataset on GitHub](https://github.com/googlecreativelab/quickdraw-dataset)

## Project Description
This project aims to explore the training and deployment of a Neural Network for image classification using the QuickDraw dataset. For now, it's a Multi-Layer Perceptron (MLP).

The focus is on:

1. Training a model with selected drawings (not the entire dataset).
2. Deploying the trained model into a web interface to interactively classify sketches.
3. The repository has a didactic purpose, serving as part of my learning process in machine learning and its creative applications.

# Repository Structure

```
quick-draw/
│
├── backend/                # Backend Flask
│   ├── app.py              # Flask app
│   ├── model_weights.pth   # Trained PyTorch model
│   └── preprocess.py       # Preprocessing functions
│
├── frontend/               # Frontend p5.js
│   ├── index.html          # HTML file
│   ├── sketch.js           # p5.js script
│   └── styles.css          # Optional styles
│
└── requirements.txt        # Python dependencies

```

- Quick-Main.ipynb:
    A Jupyter Notebook containing the full pipeline:
    - Training the MLP model
    - Evaluating the model
    - Saving the trained model

- Backend (Flask Server):
The Flask server loads the trained model and serves predictions for incoming sketch inputs.

- Frontend (P5.js Interface):
A simple web interface built using P5.js, where users can draw sketches and see the model predictions in real-time.

## Key Technologies
- Python: Model training and backend development.
- Flask: Backend server for model inference.
- P5.js: Frontend for interactive sketching.
- Jupyter Notebook: Development and experimentation.

## Purpose
This project is part of the CAS - AI for Creative Practices, where I am exploring the intersection of machine learning and creative processes. It highlights practical steps for:

- Training a machine learning model
- Deploying it interactively
- Learning and experimenting in an applied, creative context.

---

# How to Use

1. Clone this repository:
```
git clone https://github.com/linalopes/quick-draw.git
cd quick-draw
```

2. Install Python dependencies in your enviroment:
```
conda activate aicp-image
pip install -r requirements.txt
```

3. Run the Jupyter Notebook (QuickMain.ipynb) to train the MLP model and save it. The Jupyter Notebook was made on MacOS M3 (mps instead of gpu or cpu).

4. Start the Flask server:
```
python backend/app.py
```

5. Start a server for the Web-interface:
```
cd front-end
http-server
```

56. Open the P5.js interface in your browser to interact with the model.

---
# Next Steps

1. Build and Compare with a CNN

- Develop a **Convolutional Neural Network (CNN)** model using the same QuickDraw dataset.
- Compare its performance with the current Multi-Layer Perceptron (MLP) model in terms of accuracy, training time, and usability.

2. AI for Kids – Expanding the Project

- This exercise serves as a foundation for a broader project called "**AI for Kids**", aimed at making machine learning approachable and creative for young learners.
- **Future steps include**:
    - **Sketch Enhancement**: Use simple QuickDraw sketches as input and render them into more attractive illustrations.
    - **Integration with APIs**: Leverage tools like Replicate or custom pipelines to transform sketches into refined images. For example:
        - A simple sketch of a hat could be transformed into a detailed and colorful illustration.
    - **Animation Possibilities**: Extend the enhanced illustrations into animations to make the outputs even more engaging.

3. Pipeline Development

- Construct a seamless pipeline that integrates the sketch input, processing through APIs or custom models, and generating polished outputs (illustrations or animations).

Happy Coding!