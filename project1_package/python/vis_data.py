import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
import os

# Load the model architecture
layers = get_lenet()
params = init_convnet(layers)

# Load the network
data = loadmat('../results/lenet.mat')
params_raw = data['params']

for params_idx in range(len(params)):
    raw_w = params_raw[0,params_idx][0,0][0]
    raw_b = params_raw[0,params_idx][0,0][1]
    assert params[params_idx]['w'].shape == raw_w.shape, 'weights do not have the same shape'
    assert params[params_idx]['b'].shape == raw_b.shape, 'biases do not have the same shape'
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

# Load data
fullset = False
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist(fullset)
m_train = xtrain.shape[1]

batch_size = 1
layers[0]['batch_size'] = batch_size

# Select an image from the test set
img = xtest[:,0]
img = np.reshape(img, (28, 28), order='F')
plt.imshow(img.T, cmap='gray')
plt.show()

# Run the image through the network
output = convnet_forward(params, layers, xtest[:,0:1])

output_1 = np.reshape(output[0]['data'], (28,28), order='F')
plt.imshow(output_1, cmap='gray')
plt.show()

##### Fill in your code here to plot the features ######

# first layer is the data layer
# Extract outputs from CONV (second layer) and ReLU (third layer)
conv_output = output[1]['data']  # CONV layer output
relu_output = output[2]['data']  # ReLU layer output

# Get dimensions for the CONV and ReLU layer outputs
h_conv = output[1]['height']
w_conv = output[1]['width']
num_filters_conv = output[1]['channel']

h_relu = output[2]['height']
w_relu = output[2]['width']
num_filters_relu = output[2]['channel']

# Reshape the outputs
conv_output = conv_output.reshape((h_conv, w_conv, num_filters_conv), order='F')
relu_output = relu_output.reshape((h_relu, w_relu, num_filters_relu), order='F')

# Plot 20 feature maps from the CONV layer
fig_conv, axes_conv = plt.subplots(4, 5, figsize=(12, 8))
counter = 1
for i in range(4):
    for j in range(5):
        ax = axes_conv[i, j]
        if counter <= num_filters_conv:
            # Transpose the data for better visualization
            ax.imshow(conv_output[:, :, counter - 1].T, cmap='gray')
            ax.set_title(f'Filter {counter}')
        ax.axis('off')
        counter += 1
fig_conv.suptitle('Output of Conv Layer')
plt.savefig('../results/conv_layer_features.png')
plt.show()

# Process ReLU output (set positive values to 0, negative to 1 for better visualization)
relu_output_thresholded = np.where(relu_output > 0, 0, 1)

# Plot 20 feature maps from the ReLU layer
fig_relu, axes_relu = plt.subplots(4, 5, figsize=(12, 8))
counter = 1
for i in range(4):
    for j in range(5):
        ax = axes_relu[i, j]
        if counter <= num_filters_relu:
            # Transpose the data
            ax.imshow(relu_output_thresholded[:, :, counter - 1].T, cmap='gray')
            ax.set_title(f'Filter {counter}')
        ax.axis('off')
        counter += 1
fig_relu.suptitle('Activated pixels from ReLU layer')
plt.savefig('../results/relu_layer_features.png')
plt.show()