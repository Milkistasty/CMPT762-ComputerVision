import numpy as np
import matplotlib.pyplot as plt
from eightpoint import eightpoint
from displayEpipolarF import displayEpipolarF

correspondences_path = '../data/someCorresp.npy'  
image1_path = '../data/im1.png'  
image2_path = '../data/im2.png' 

# Load point correspondences
correspondences = np.load(correspondences_path, allow_pickle=True).item()

pts1 = correspondences['pts1']
pts2 = correspondences['pts2']
M = correspondences['M']

img1 = plt.imread(image1_path)
img2 = plt.imread(image2_path)

print(f"Normalization factor M: {M}")

# Compute the Fundamental matrix using the Eight-Point Algorithm
F = eightpoint(pts1, pts2, M)

print("Computed Fundamental Matrix F:")
print(F)

# Visualize the epipolar lines using displayEpipolarF
displayEpipolarF(img1, img2, F)

