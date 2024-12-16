import os
import io
import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Import Flask-CORS
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
from preprocess import preprocess_image  # Custom preprocess function

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

# Load trained model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Mapping of indices to class names
label_dict = {0: 'apple', 1: 'book', 2: 'cannon', 3: 'crayon', 4: 'eye',
              5: 'face', 6: 'flower', 7: 'nail', 8: 'pear', 9: 'piano',
              10: 'radio', 11: 'spider', 12: 'star', 13: 'sun'}

# Define the model architecture
def build_model(input_size, output_size, hidden_sizes, dropout=0.0):
    """
    Build the feedforward neural network with Batch Normalization and Dropout.
    """
    model = torch.nn.Sequential(OrderedDict([
        ('fc1', torch.nn.Linear(input_size, hidden_sizes[0])),
        ('relu1', torch.nn.ReLU()),
        ('bn1', torch.nn.BatchNorm1d(hidden_sizes[0])),
        ('dropout1', torch.nn.Dropout(dropout)),

        ('fc2', torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ('relu2', torch.nn.ReLU()),
        ('bn2', torch.nn.BatchNorm1d(hidden_sizes[1])),
        ('dropout2', torch.nn.Dropout(dropout)),

        ('fc3', torch.nn.Linear(hidden_sizes[1], hidden_sizes[2])),
        ('relu3', torch.nn.ReLU()),
        ('bn3', torch.nn.BatchNorm1d(hidden_sizes[2])),
        ('dropout3', torch.nn.Dropout(dropout)),

        ('logits', torch.nn.Linear(hidden_sizes[2], output_size))
    ]))
    return model


model_path = 'model_weights.pth'
input_size = 784
output_size = 14  # Total classes
hidden_sizes = [512, 256, 128]

model = build_model(input_size, output_size, hidden_sizes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

@app.route("/preprocess", methods=["POST"])
def preprocess_and_return_image():
    """
    Endpoint to preprocess an image and return the processed version.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Load the uploaded image
    image_file = request.files["image"]
    image = Image.open(io.BytesIO(image_file.read()))
    
    # Apply preprocessing
    processed_image = preprocess_image(image, padding=25)

    # Convert the tensor back to a PIL image for visualization
    processed_image = processed_image.squeeze(0)  # Remove batch dimension
    processed_image = (processed_image * 0.5 + 0.5) * 255  # Denormalize to [0, 255]
    processed_image = processed_image.cpu().numpy().astype(np.uint8)  # Convert to NumPy
    processed_image_pil = Image.fromarray(processed_image, mode="L")

    # Return the image as a response
    img_io = io.BytesIO()
    processed_image_pil.save(img_io, "PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to predict the class of an uploaded image.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Load and preprocess image
    image_file = request.files["image"]
    image = Image.open(io.BytesIO(image_file.read()))
    processed_image = preprocess_image(image, padding=25).to(device)

    # Model prediction
    with torch.no_grad():
        logits = model(processed_image.view(1, -1))  # Flatten image to 784 for MLP
        probabilities = F.softmax(logits, dim=1)     # Get probabilities
        top3_probs, top3_classes = torch.topk(probabilities, 3)  # Top-3 predictions

    # Map indices to class names using label_dict
    predictions = [{"class": label_dict[top3_classes[0][i].item()], 
                    "probability": round(top3_probs[0][i].item() * 100, 2)} 
                   for i in range(3)]

    return jsonify(predictions)


# Run the Flask server
if __name__ == "__main__":
    app.run(debug=True)
