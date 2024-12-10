from bounding_box_corners import *
import matplotlib.pyplot as plt
from depth_map_with_multi_view_stereo import *

def load_depth_map(filename):
    depth_map = np.load(filename)
    return depth_map

def depth_map_to_point_cloud(depth_map, K_inv, R_inv, t):
    h, w = depth_map.shape
    points_3D = []
    colors = []
    for y in range(h):
        for x in range(w):
            d = depth_map[y, x]
            if d > 0:
                X = get_3d_coord(x, y, d, K_inv, R_inv, t)
                points_3D.append(X)
                # Optionally, collect colors from the image
                # colors.append(I0[y, x])
    points_3D = np.array(points_3D)
    return points_3D

def save_point_cloud_to_obj(points_3D, filename):
    with open(filename, 'w') as f:
        for pt in points_3D:
            f.write(f"v {pt[0]} {pt[1]} {pt[2]}\n")
    print(f"Saved point cloud to {filename}")

# Load depth map
depth_map = load_depth_map('../results/depth_map.npy')

# Load camera parameters
camera_params = load_camera_parameters('../data/templeR_par.txt')

# Reference image I0
I0_name = 'templeR0013.png'
K0 = camera_params[I0_name]['K']
R0 = camera_params[I0_name]['R']
t0 = camera_params[I0_name]['t']

K0_inv = np.linalg.inv(K0)
R0_inv = np.linalg.inv(R0)

# Convert depth map to point cloud
points_3D = depth_map_to_point_cloud(depth_map, K0_inv, R0_inv, t0)

# Save point cloud to OBJ file
save_point_cloud_to_obj(points_3D, '../results/point_cloud.obj')

