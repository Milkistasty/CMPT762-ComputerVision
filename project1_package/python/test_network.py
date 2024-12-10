import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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


# Testing the network
#### Modify the code to get the confusion matrix ####
all_preds = []
all_labels = []

batch_size = 100
layers[0]['batch_size'] = batch_size

for i in range(0, xtest.shape[1], batch_size):
    end = min(i + batch_size, xtest.shape[1])
    _, P = convnet_forward(params, layers, xtest[:, i:end], test=True)
    preds = np.argmax(P, axis=0)
    all_preds.extend(preds)
    all_labels.extend(ytest[0, i:end])

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# hint: 
#     you can use confusion_matrix from sklearn.metrics (pip install -U scikit-learn)
#     to compute the confusion matrix. Or you can write your own code :)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Compute accuracy, recall, precision, and F1 score
report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)])
print(report)

# Display accuracy and recall per class using classification report
classification_report_df = pd.DataFrame(classification_report(all_labels, all_preds, output_dict=True)).transpose()

# Plot classification report as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(classification_report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f')
plt.title('Classification Report - Precision, Recall, F1-Score')
plt.show()

# Identify the top two confused pairs
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
top_two_confused = np.unravel_index(np.argsort(cm_no_diag, axis=None)[-2:], cm_no_diag.shape)
print(f"Top two confused pairs: {list(zip(top_two_confused[0], top_two_confused[1]))}")