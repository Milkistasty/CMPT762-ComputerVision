import numpy as np
import os
from PIL import Image
from scipy.io import loadmat
from utils import get_lenet
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt

# Load the model architecture
layers = get_lenet()
params = init_convnet(layers)

# Load the pre-trained model
data = loadmat('../results/lenet.mat')
params_raw = data['params']

for params_idx in range(len(params)):
    params[params_idx]['w'] = params_raw[0, params_idx][0, 0][0]
    params[params_idx]['b'] = params_raw[0, params_idx][0, 0][1]

# Process the image
def process_image(img_path):
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    img_resized = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img_resized, dtype=np.float32) / 255.0  # Normalize
    img_flattened = img_array.flatten()
    return img, img_array, img_flattened  # Return original, processed, and flattened images

def get_image_paths(directory):
    image_paths = []
    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Only process images
            image_paths.append(os.path.join(directory, filename))
    
    assert len(image_paths)>0, "No files in this directory: " + directory
    
    return image_paths

# Directory where the real-world images are stored
image_directory = "../images/real_world_examples/"

# Get the image file paths
image_paths = get_image_paths(image_directory)

# Process and prepare images for the model
imgs = np.zeros((28*28, len(image_paths)))
for i, img_path in enumerate(image_paths):
    original_img, processed_img, flattened_img = process_image(img_path)
    imgs[:, i] = flattened_img

    # Show the original, processed, and flattened images in the same window
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    # original image
    ax[0].imshow(original_img, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    # processed image
    ax[1].imshow(processed_img, cmap='gray')
    ax[1].set_title("Processed Image (28x28)")
    ax[1].axis('off')
    
    # flattened image
    ax[2].imshow(flattened_img.reshape(28, 28), cmap='gray')
    ax[2].set_title("Flattened Image (28x28)")
    ax[2].axis('off')
    
    plt.suptitle(f"Image: {os.path.basename(img_path)}")
    plt.show()

    input("Press Enter to continue to the next image...")

layers[0]['batch_size'] = len(image_paths)

# Make predictions
_, P = convnet_forward(params, layers, imgs, test=True)

# Get predicted labels (index of the maximum probability)
predicted_labels = np.argmax(P, axis=0)

# Display predictions
for i, img_path in enumerate(image_paths):
    print(f"Image: {os.path.basename(img_path)}, Predicted Label: {predicted_labels[i]}")
