import numpy as np

def get_disparity(im1, im2, maxDisp, windowSize):
    """
    creates a disparity map from a pair of rectified images im1 and
    im2, given the maximum disparity MAXDISP and the window size WINDOWSIZE.
    """
    h, w = im1.shape
    dispM = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            min_ssd = float('inf')
            best_disp = 0
            for d in range(maxDisp):
                if x - d >= 0:
                    ssd = compute_ssd(im1, im2, y, x, d, windowSize)
                    if ssd < min_ssd:
                        min_ssd = ssd
                        best_disp = d
            dispM[y, x] = best_disp

    return dispM

def compute_ssd(im1, im2, y, x, d, windowSize):
    half_window = windowSize // 2
    h, w = im1.shape
    
    # Define the window in im1
    y_min = max(y - half_window, 0)
    y_max = min(y + half_window, h - 1)
    x_min = max(x - half_window, 0)
    x_max = min(x + half_window, w - 1)
    window1 = im1[y_min:y_max + 1, x_min:x_max + 1]
    
    # Define the window in im2 shifted by disparity d
    x_shifted = x - d
    x2_min = max(x_shifted - half_window, 0)
    x2_max = min(x_shifted + half_window, w - 1)
    window2 = im2[y_min:y_max + 1, x2_min:x2_max + 1]
    
    # Adjust windows to the same size if necessary
    h1, w1 = window1.shape
    h2, w2 = window2.shape
    h_min = min(h1, h2)
    w_min = min(w1, w2)
    window1 = window1[:h_min, :w_min]
    window2 = window2[:h_min, :w_min]
    
    # Compute SSD
    ssd = np.sum((window1.astype(np.float32) - window2.astype(np.float32)) ** 2)
    return ssd