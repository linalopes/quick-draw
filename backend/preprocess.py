import numpy as np
from PIL import Image
from torchvision import transforms

def preprocess_image(image, padding=25):
    """
    Preprocess the input image:
    - Grayscale conversion
    - Bounding box cropping with padding
    - Inversion of colors (white -> black, black -> white)
    - Resize to 28x28
    - Normalize to [-1, 1]
    """
    # Convert image to grayscale
    image = image.convert('L')  # Grayscale

    # Convert image to NumPy array
    image_array = np.array(image)

    # Extract bounding box of content
    non_white_mask = image_array < 255  # Pixels with values < 255 are considered content
    coords = np.argwhere(non_white_mask)

    if coords.size > 0:  # Check if there is content
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        # Add padding
        x_min = max(x_min - padding, 0)
        y_min = max(y_min - padding, 0)
        x_max = min(x_max + padding, image_array.shape[0])
        y_max = min(y_max + padding, image_array.shape[1])

        # Crop the image
        cropped_image = image.crop((y_min, x_min, y_max, x_max))
    else:
        cropped_image = image

    # Invert the colors: white -> black, black -> white
    cropped_array = np.array(cropped_image)
    inverted_array = np.where(cropped_array == 255, 0, 255).astype(np.uint8)
    inverted_image = Image.fromarray(inverted_array)

    # Resize and normalize
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),         # Resize to 28x28
        transforms.ToTensor(),               # Convert to tensor
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
    ])

    return preprocess(inverted_image)

