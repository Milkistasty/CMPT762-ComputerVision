import numpy as np

def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    """
    creates a depth map from a disparity map (DISPM).
    """
    # Compute camera centers c1 and c2
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)
    baseline = np.linalg.norm(c2 - c1)
    f = K1[0, 0]  # Assuming fx is at (0,0) in K1
    
    # Compute depth map
    depthM = np.zeros_like(dispM, dtype=np.float32)
    valid_disp = dispM > 0
    depthM[valid_disp] = (baseline * f) / dispM[valid_disp]
    
    return depthM

