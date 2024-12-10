import numpy as np

def rectify_pair(K1, K2, R1, R2, t1, t2):
    """
    takes left and right camera paramters (K, R, T) and returns left
    and right rectification matrices (M1, M2) and updated camera parameters. You
    can test your function using the provided script testRectify.py
    """
    # YOUR CODE HERE
    # 1. Compute optical centers c1 and c2
    c1 = -R1.T @ t1.reshape(3)
    c2 = -R2.T @ t2.reshape(3)

    # 2. Compute New rotation matrix r1, r2, r3
    r1 = c1 - c2
    r1 = r1 / np.linalg.norm(r1)

    r2 = np.cross(R1[2, :], r1)
    r2 = r2 / np.linalg.norm(r2)

    r3 = np.cross(r2, r1)

    # 3. Compute New Rotation Matrix R_new
    R_new = np.vstack((r1, r2, r3)).T  # Shape (3, 3)

    # 4. Compute New Intrinsic parameter Kn
    Kn = K2.copy()

    # 5. Compute New Translation Vectors
    t1n = -R_new @ c1.reshape(3, 1)
    t2n = -R_new @ c2.reshape(3, 1)

    # 6. Compute Updated Rotation Matrices R1n and R2n
    R1n = R_new
    R2n = R_new

    # 7. Compute Rectification Matrices M1 and M2
    M1 = Kn @ R_new @ np.linalg.inv(K1 @ R1)
    M2 = Kn @ R_new @ np.linalg.inv(K2 @ R2)

    # 8. Compute Updated Intrinsic Matrices K1n and K2n
    K1n = K1.copy()
    K2n = K2.copy()

    return M1, M2, K1n, K2n, R1n, R2n, t1n, t2n

