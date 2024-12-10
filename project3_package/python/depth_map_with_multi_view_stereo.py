from bounding_box_corners import *
import matplotlib.pyplot as plt

def compute_depth_range(corners, P):
    num_points = corners.shape[0]
    homogeneous_3D = np.hstack((corners, np.ones((num_points, 1))))
    projected_points = P @ homogeneous_3D.T  # Shape: 3 x N
    depths = projected_points[2, :]  # Third coordinate is depth
    min_depth = depths.min()
    max_depth = depths.max()
    return min_depth, max_depth

def get_3d_coord(x, y, d, K_inv, R_inv, t):
    pixel_homogeneous = np.array([x, y, 1])
    cam_coords = K_inv @ pixel_homogeneous * d
    world_coords = R_inv @ (cam_coords - t.flatten())
    return world_coords

def compute_consistency(I0, I1, X, P0, P1):
    # Project points into I0
    points_2D_I0 = project_points(X, P0)
    # Collect colors from I0
    C0 = sample_colors(I0, points_2D_I0)
    
    # Project points into I1
    points_2D_I1 = project_points(X, P1)
    # Collect colors from I1
    C1 = sample_colors(I1, points_2D_I1)
    
    # Compute NCC
    ncc_score = normalized_cross_correlation(C0, C1)
    return ncc_score

def sample_colors(image, points_2D):
    h, w, _ = image.shape
    colors = []
    for pt in points_2D:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < w and 0 <= y < h:
            colors.append(image[y, x])
        else:
            colors.append([0, 0, 0])  # Assign black color if out of bounds
    return np.array(colors)

def normalized_cross_correlation(C0, C1):
    # Flatten to 1D arrays
    C0 = C0.reshape(-1)
    C1 = C1.reshape(-1)
    
    # Subtract mean
    C0_mean = C0 - np.mean(C0)
    C1_mean = C1 - np.mean(C1)
    
    # Normalize
    C0_norm = C0_mean / np.linalg.norm(C0_mean) if np.linalg.norm(C0_mean) != 0 else C0_mean
    C1_norm = C1_mean / np.linalg.norm(C1_mean) if np.linalg.norm(C1_mean) != 0 else C1_mean
    
    # Compute dot product
    ncc = np.dot(C0_norm, C1_norm)
    return ncc

def compute_depth_map(I0, other_images, camera_params, P0, K0, R0, t0, min_depth, max_depth, depth_step, window_size=7):
    h, w, _ = I0.shape
    depth_map = np.zeros((h, w))
    
    K0_inv = np.linalg.inv(K0)
    R0_inv = np.linalg.inv(R0)
    
    offsets = np.arange(-(window_size // 2), window_size // 2 + 1)
    
    for y in range(0, h, 5): 
        for x in range(0, w, 5):
            # Skip background pixels
            if np.all(I0[y, x] < 10):
                continue
            best_ncc = -np.inf
            best_depth = 0
            for d in np.arange(min_depth, max_depth, depth_step):
                # Collect 3D points in the window
                points_3D = []
                for dy in offsets:
                    for dx in offsets:
                        x_q = x + dx
                        y_q = y + dy
                        if 0 <= x_q < w and 0 <= y_q < h:
                            X_q = get_3d_coord(x_q, y_q, d, K0_inv, R0_inv, t0)
                            points_3D.append(X_q)
                points_3D = np.array(points_3D)
                
                # Compute consistency scores with other images
                ncc_scores = []
                for img_name in other_images:
                    I_i = other_images[img_name]
                    P_i = camera_params[img_name]['P']
                    ncc = compute_consistency(I0, I_i, points_3D, P0, P_i)
                    ncc_scores.append(ncc)
                avg_ncc = np.mean(ncc_scores)
                # Update best depth
                if avg_ncc > best_ncc:
                    best_ncc = avg_ncc
                    best_depth = d
            # Set depth value if best NCC is above threshold
            if best_ncc > 0.7:
                depth_map[y, x] = best_depth
    return depth_map

def visualize_depth_map(depth_map):
    depthM_display = np.where(depth_map > 0, depth_map, np.nan)
    depthM_display = (depthM_display - np.nanmin(depthM_display)) / (np.nanmax(depthM_display) - np.nanmin(depthM_display))
    depthM_display = 1 - depthM_display

    plt.figure(figsize=(8, 8))
    plt.imshow(depthM_display, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.gca().set_facecolor('black')
    plt.savefig('../results/depth_white_on_black.png', dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.show()


camera_params = load_camera_parameters('../data/templeR_par.txt')
images = load_images('../data', camera_params)

# Reference image I0
I0_name = 'templeR0013.png'
I0 = images[I0_name]
P0 = camera_params[I0_name]['P']
K0 = camera_params[I0_name]['K']
R0 = camera_params[I0_name]['R']
t0 = camera_params[I0_name]['t']

# Other images
other_image_names = ['templeR0014.png', 'templeR0016.png', 'templeR0043.png', 'templeR0045.png']
other_images = {name: images[name] for name in other_image_names}

# Compute bounding box corners
corners = compute_bounding_box_corners()

# Compute depth range
min_depth, max_depth = compute_depth_range(corners, P0)
depth_step = 0.05

print(f"Depth range: {min_depth:.4f} to {max_depth:.4f}, step: {depth_step:.4f}")

# Compute depth map
depth_map = compute_depth_map(I0, other_images, camera_params, P0, K0, R0, t0, min_depth, max_depth, depth_step)

# Visualize depth map
visualize_depth_map(depth_map)

# Save depth map
np.save('../results/depth_map.npy', depth_map)