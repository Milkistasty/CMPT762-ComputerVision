import numpy as np
import os
import sys
from PIL import Image
from scipy.io import loadmat
import cv2
import matplotlib.pyplot as plt

# Load the model architecture
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
python_dir = os.path.join(parent_dir, 'python')
sys.path.append(python_dir)

from utils import get_lenet
from conv_net import convnet_forward
from init_convnet import init_convnet

layers = get_lenet()
params = init_convnet(layers)

# Load the pre-trained model
data = loadmat(os.path.join(parent_dir, 'results', 'lenet.mat'))
params_raw = data['params']

for params_idx in range(len(params)):
    params[params_idx]['w'] = params_raw[0, params_idx][0, 0][0]
    params[params_idx]['b'] = params_raw[0, params_idx][0, 0][1]

# Function to process an image and extract digits using contours and bounding boxes
def process_image(img_path, threshold_val=150, stre=1):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Ensure stre is at least 1 to avoid kernel size errors
    if stre < 1:
        stre = 1

    # Apply morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stre, stre))
    background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Invert and threshold the image
    com = cv2.bitwise_not(background)
    _, com = cv2.threshold(com, threshold_val, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels_im = cv2.connectedComponents(com)

    # Extract bounding boxes for each component
    bounding_boxes = []
    for label in range(1, num_labels):  # Skip background
        component_mask = (labels_im == label).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(component_mask)
        bounding_boxes.append((x, y, w, h))

    # Prepare digits for the network
    digits = []
    for x, y, w, h in bounding_boxes:
        digit = com[y:y+h, x:x+w]

        # Pad to ensure the digit is square
        aspect_ratio = w / h
        if aspect_ratio > 1:
            pad_height = (w - h) // 2
            digit = cv2.copyMakeBorder(digit, pad_height, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            pad_width = (h - w) // 2
            digit = cv2.copyMakeBorder(digit, 0, 0, pad_width, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Resize to 28x28
        digit_resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_LANCZOS4)
        digit_flattened = digit_resized.flatten() / 255.0  # Normalize to [0, 1]
        digits.append(digit_flattened)

    return digits, bounding_boxes, com

# Get image paths
def get_image_paths(directory):
    image_paths = []
    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg", ".JPG")):
            image_paths.append(os.path.join(directory, filename))
    return image_paths

image_directory = os.path.join(parent_dir, "images")

# Get the image file paths
image_paths = get_image_paths(image_directory)

# Process each image and make predictions
for img_path in image_paths:
    img_name = os.path.basename(img_path)

    # Adjust threshold and structuring element size for each image
    if img_name == 'image4.jpg':
        threshold = 40
        stre = 1
    elif img_name == 'image3.png':
        threshold = 100
        stre = 1  # Avoid stre = 0 to prevent kernel errors
    else:
        threshold = 150
        stre = 1

    digits, bounding_boxes, img_bin = process_image(img_path, threshold_val=threshold, stre=stre)

    imgs = np.stack(digits, axis=1)

    layers[0]['batch_size'] = imgs.shape[1]

    # Make predictions
    _, P = convnet_forward(params, layers, imgs, test=True)

    # Get predicted labels
    predicted_labels = np.argmax(P, axis=0)

    # Display bounding boxes and predictions with larger window and appropriately scaled text
    plt.figure(figsize=(32, 32))  # Increase figure size
    img_with_boxes = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)

    for i, (x, y, w, h) in enumerate(bounding_boxes):
        # Draw bounding box
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Plot the predicted digit below the bounding box with adjusted font size
        # Adjust threshold and structuring element size for each image
        if img_name == 'image4.jpg':
            text_x = x - 3
            text_y = y
        elif img_name == 'image3.png':
            text_x = x - 3
            text_y = y
        else:
            text_x = x
            text_y = y + 15

        font_scale = 0.4 if img_name == 'image4.jpg' or img_name == 'image3.png' else 1.5  # Adjust font size based on image size

        cv2.putText(img_with_boxes, str(predicted_labels[i]), (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

    # Show the image with bounding boxes and predictions
    plt.imshow(img_with_boxes)
    plt.title(f"Image: {img_name} - Predictions")
    plt.axis('off')
    plt.show()
