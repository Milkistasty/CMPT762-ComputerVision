import numpy as np
import cv2

def epipolarCorrespondence(img1, img2, F, pts1):
    """
    Args:
        img1:    Image 1
        img2:    Image 2
        F:      Fundamental Matrix from img1 to img2
        pts1:   coordinates of points in image 1
    Returns:
        pts2:   coordinates of points in image 2
    """
    
    N = pts1.shape[0]
    pts2 = np.zeros_like(pts1)
    height, width = img1.shape[0], img1.shape[1]
    window_size = 5  # Half window size (patch: 11x11)
    max_offset = 70  # Max search range along the epipolar line

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    for i in range(N):
        x1, y1 = pts1[i]
        p1 = np.array([x1, y1, 1])

        # Compute the epipolar line in image 2
        eline = F @ p1  # l' = F * p
        # Normalize the line
        eline = eline / np.sqrt(eline[0]**2 + eline[1]**2)

        # Define the window around the point in img1
        x1_min = int(max(0, x1 - window_size))
        x1_max = int(min(width - 1, x1 + window_size))
        y1_min = int(max(0, y1 - window_size))
        y1_max = int(min(height - 1, y1 + window_size))

        window1 = img1_gray[y1_min:y1_max+1, x1_min:x1_max+1]

        min_error = float('inf')
        best_x2, best_y2 = None, None

        # Search along the epipolar line in img2
        y_start = max(0, y1 - max_offset)
        y_end = min(height, y1 + max_offset)

        for y2 in range(y_start, y_end):
            # Solve for x2 given y2: a*x2 + b*y2 + c = 0
            a, b, c = eline
            if abs(a) > 1e-6:
                x2 = int(round((-b * y2 - c) / a))
                if x2 - window_size >= 0 and x2 + window_size < width and y2 - window_size >= 0 and y2 + window_size < height:
                    x2_min = x2 - window_size
                    x2_max = x2 + window_size
                    y2_min = y2 - window_size
                    y2_max = y2 + window_size

                    window2 = img2_gray[y2_min:y2_max+1, x2_min:x2_max+1]

                    if window1.shape == window2.shape:
                        # Compute Sum of Squared Differences (SSD)
                        error = np.sum((window1 - window2) ** 2)
                        if error < min_error:
                            min_error = error
                            best_x2, best_y2 = x2, y2
            else:
                # If line is vertical, skip this point
                continue

        if best_x2 is not None and best_y2 is not None:
            pts2[i] = [best_x2, best_y2]
        else:
            # If no match is found, return the original point or set to -1
            pts2[i] = [x1, y1]  # Or pts2[i] = [-1, -1]

    return pts2