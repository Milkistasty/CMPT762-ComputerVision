import numpy as np
from epipolarCorrespondence import *
from eightpoint import *
from epipolarMatchGUI import *
import cv2

correspondences_path = '../data/someCorresp.npy'  
image1_path = '../data/im1.png'  
image2_path = '../data/im2.png'  

correspondences = np.load(correspondences_path, allow_pickle=True).item()

pts1 = correspondences['pts1']
pts2 = correspondences['pts2']
M = correspondences['M']

img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

F = eightpoint(pts1, pts2, M)
print("Computed Fundamental Matrix F:")
print(F)

# Run the Epipolar Match GUI
coords1, coords2 = epipolarMatchGUI(img1, img2, F)