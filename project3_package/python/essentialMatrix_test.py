import numpy as np
import matplotlib.pyplot as plt
from essentialMatrix import *
from eightpoint import *
from epipolarMatchGUI import *
import cv2


correspondences_path = '../data/someCorresp.npy'  
intrinsics_path = '../data/intrinsics.npy'        
image1_path = '../data/im1.png'                   
image2_path = '../data/im2.png'                   

correspondences = np.load(correspondences_path, allow_pickle=True).item()
pts1 = correspondences['pts1']  
pts2 = correspondences['pts2']  
M = correspondences['M']        

intrinsics = np.load(intrinsics_path, allow_pickle=True).item()
K1 = intrinsics['K1']  # 3x3 matrix
K2 = intrinsics['K2']  # 3x3 matrix

img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

F = eightpoint(pts1, pts2, M)


E = essentialMatrix(F, K1, K2)
print("Computed Essential Matrix E:")
print(E)

# Scale the Essential Matrix
scale = np.array([[1/M, 0, 0],
                    [0, 1/M, 0],
                    [0, 0, 1]])
E_scaled = scale.T @ E @ scale
print("Scaled Essential Matrix E:")
print(E_scaled)


# Run the Epipolar Match GUI with the scaled Essential Matrix
print("Launching Epipolar Match GUI with Essential Matrix...")
coordsIM1, coordsIM2 = epipolarMatchGUI(img1, img2, E_scaled)
print("Selected Correspondences:")
print("Image 1 Points:\n", coordsIM1)
print("Image 2 Points:\n", coordsIM2)