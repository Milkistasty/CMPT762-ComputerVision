import numpy as np
import cv2
import os

def compute_bounding_box_corners():
    min_coords = np.array([-0.023121, -0.038009, -0.091940])
    max_coords = np.array([0.078626, 0.121636, -0.017395])

    # Generate all combinations of min and max coordinates
    corners = np.array(np.meshgrid(
        [min_coords[0], max_coords[0]],
        [min_coords[1], max_coords[1]],
        [min_coords[2], max_coords[2]]
    )).T.reshape(-1, 3)
    return corners

def load_camera_parameters(filename):
    camera_params = {}
    with open(filename, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            img_name = tokens[0]
            # Intrinsic parameters K
            k_values = np.array(tokens[1:10], dtype=float).reshape(3, 3)
            # Rotation matrix R
            r_values = np.array(tokens[10:19], dtype=float).reshape(3, 3)
            # Translation vector t
            t_values = np.array(tokens[19:22], dtype=float).reshape(3, 1)
            # Projection matrix P = K * [R | t]
            RT = np.hstack((r_values, t_values))
            P = k_values @ RT
            camera_params[img_name] = {
                'K': k_values,
                'R': r_values,
                't': t_values,
                'P': P
            }
    return camera_params

def load_images(image_folder, camera_params):
    images = {}
    for img_name in camera_params.keys():
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            images[img_name] = img
        else:
            print(f"Image {img_name} not found.")
    return images

def project_points(points_3D, P):
    num_points = points_3D.shape[0]
    homogeneous_3D = np.hstack((points_3D, np.ones((num_points, 1))))
    projected_points = P @ homogeneous_3D.T  # Shape: 3 x N
    projected_points /= projected_points[2, :]  # Normalize by the third coordinate
    points_2D = projected_points[:2, :].T  # Shape: N x 2
    return points_2D

def visualize_projected_corners(images, camera_params, corners):
    output_folder = '../results/projected_corners'
    os.makedirs(output_folder, exist_ok=True)
    for img_name, img in images.items():
        img_copy = img.copy()
        P = camera_params[img_name]['P']
        projected_points = project_points(corners, P)
        for pt in projected_points:
            x, y = int(round(pt[0])), int(round(pt[1]))
            # Draw a circle at the projected point
            cv2.circle(img_copy, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
        # Save the image
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, img_copy)
        print(f"Saved projected corners on {img_name} to {output_path}")


corners = compute_bounding_box_corners()
camera_params = load_camera_parameters('../data/templeR_par.txt')
images = load_images('../data', camera_params)
visualize_projected_corners(images, camera_params, corners)