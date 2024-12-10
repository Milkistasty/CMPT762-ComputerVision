import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from PIL import Image
from eightpoint import eightpoint
from epipolarCorrespondence import epipolarCorrespondence
from essentialMatrix import essentialMatrix
from camera2 import camera2
from triangulate import triangulate
from displayEpipolarF import displayEpipolarF
from epipolarMatchGUI import epipolarMatchGUI
import os
from mpl_toolkits.mplot3d import Axes3D

data_dir = '../data'  
result_dir = '../results' 

some_corresp_path = os.path.join(data_dir, 'someCorresp.npy')
intrinsics_path = os.path.join(data_dir, 'intrinsics.npy')
image1_path = os.path.join(data_dir, 'im1.png')
image2_path = os.path.join(data_dir, 'im2.png')
temple_coords_path = os.path.join(data_dir, 'templeCoords.npy')

correspondences = np.load(some_corresp_path, allow_pickle=True).item()
pts1 = correspondences['pts1']
pts2 = correspondences['pts2']
M = correspondences['M']

intrinsics = np.load(intrinsics_path, allow_pickle=True).item()
K1 = intrinsics['K1']  # 3x3 intrinsic matrix for image 1
K2 = intrinsics['K2']  # 3x3 intrinsic matrix for image 2

img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

# Step 1: Compute F using eightpoint algorithm
F = eightpoint(pts1, pts2, M)

# Step 2: Load points from templeCoords.npy
temple_coords = np.load(temple_coords_path, allow_pickle=True).item()

pts1_temple = temple_coords['pts1'] # (N, 2) array

# Step 3: Compute corresponding points in image 2 using epipolarCorrespondence
print("Computing correspondences using epipolarCorrespondence...")
pts2_temple = epipolarCorrespondence(img1, img2, F, pts1_temple)

# Step 4: Compute E using essentialMatrix
E = essentialMatrix(F, K1, K2)
print("Computed Essential Matrix E:")
print(E)

# Step 5: Compute P1 and possible P2s using camera2
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))  # P1 = K1 * [I|0]
M2s = camera2(E)  # Returns 4 possible [R|t] matrices

# Step 6 and 7: Triangulate points and select the correct P2
max_positive_depth = 0
best_P2 = None
best_M2 = None
best_pts3d = None

for i in range(4):
    # Extract the i-th candidate for M2
    M2 = M2s[:, :, i]
    P2 = K2 @ M2  # P2 = K2 * [R|t]
    
    # Triangulate points
    pts3d = triangulate(P1, pts1_temple, P2, pts2_temple)
    
    # Check the number of points in front of both cameras
    # For camera 1, since it's at [I|0], the depth is Z coordinate
    pts3d_hom = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    depths1 = (P1 @ pts3d_hom.T)[2]
    depths2 = (P2 @ pts3d_hom.T)[2]
    
    num_positive_depth = np.sum((depths1 > 0) & (depths2 > 0))
    print(f"M2 candidate {i}: Number of points with positive depth: {num_positive_depth}")
    
    if num_positive_depth > max_positive_depth:
        max_positive_depth = num_positive_depth
        best_P2 = P2
        best_M2 = M2
        best_pts3d = pts3d

if best_P2 is None:
    print("No valid P2 found with positive depth for all points.")
    exit()

# Step 8: Compute Reprojection Errors
print("Computing Reprojection Errors...")
# Project the 3D points back to both images
pts3d_hom = np.hstack((best_pts3d, np.ones((best_pts3d.shape[0], 1))))  # Nx4
reproj1 = (P1 @ pts3d_hom.T).T  # Nx3
reproj2 = (best_P2 @ pts3d_hom.T).T  # Nx3

# Normalize to convert from homogeneous to 2D
reproj1 = reproj1[:, :2] / reproj1[:, 2, np.newaxis]
reproj2 = reproj2[:, :2] / reproj2[:, 2, np.newaxis]

# Compute Euclidean distances
errors1 = np.linalg.norm(pts1_temple - reproj1, axis=1)
errors2 = np.linalg.norm(pts2_temple - reproj2, axis=1)

mean_error1 = np.mean(errors1)
mean_error2 = np.mean(errors2)

print(f"Mean Reprojection Error in Image 1: {mean_error1:.4f} pixels")
print(f"Mean Reprojection Error in Image 2: {mean_error2:.4f} pixels")

# Step 9: Visualization of the 3D points

# Ensure result directory exists
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Define different viewing angles (elevation, azimuth)
angles = [
    (20, -60),  # View 1
    (30, 30),   # View 2
    (50, 110)   # View 3
]

# Plot and save the views
for i, (elev, azim) in enumerate(angles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(best_pts3d[:,0], best_pts3d[:,1], best_pts3d[:,2], c='r', marker='o', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Reconstruction View {i+1}')
    ax.view_init(elev=elev, azim=azim)
    
    # Save the figure
    output_path = os.path.join(result_dir, f'3D_Reconstruction_View_{i+1}.png')
    plt.savefig(output_path)
    print(f"Saved 3D reconstruction image: {output_path}")
    plt.close(fig)

# Step 10: Save extrinsic parameters for dense reconstruction
R1 = np.eye(3)
t1 = np.zeros((3,1))

R2 = best_M2[:, :3]
t2 = best_M2[:, 3].reshape(3,1)

extrinsics = {'R1': R1, 't1': t1, 'R2': R2, 't2': t2}
np.save(os.path.join(result_dir, 'extrinsics.npy'), extrinsics)
print(f"Saved extrinsic parameters to '{os.path.join(result_dir, 'extrinsics.npy')}'.")
