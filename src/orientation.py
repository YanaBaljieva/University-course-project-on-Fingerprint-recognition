import math
import numpy as np
import cv2 as cv


def calculate_angles(im, W=16, smooth=False):
    
    im = im.astype(np.float32)
    rows, cols = im.shape

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = sobel_x.T

    Gx = cv.filter2D(im, -1, sobel_x)
    Gy = cv.filter2D(im, -1, sobel_y)

    angle_rows = math.ceil(rows / W)
    angle_cols = math.ceil(cols / W)
    angles = np.zeros((angle_rows, angle_cols), dtype=np.float32)

    for r in range(angle_rows):
        for c in range(angle_cols):
            r0 = r * W
            c0 = c * W
            block_Gx = Gx[r0:min(r0 + W, rows), c0:min(c0 + W, cols)]
            block_Gy = Gy[r0:min(r0 + W, rows), c0:min(c0 + W, cols)]

            Vx = 2 * np.sum(block_Gx * block_Gy)
            Vy = np.sum(block_Gx**2 - block_Gy**2)

            if abs(Vx) > 1e-8 or abs(Vy) > 1e-8:
                angles[r, c] = 0.5 * np.arctan2(Vx, Vy)
            else:
                angles[r, c] = 0.0

    if smooth:
        angles = smooth_angles(angles)

    return angles


def smooth_angles(angles):
    cos2 = np.cos(2 * angles)
    sin2 = np.sin(2 * angles)

    kernel = cv.getGaussianKernel(5, 1)
    kernel = kernel @ kernel.T

    cos2 = cv.filter2D(cos2, -1, kernel)
    sin2 = cv.filter2D(sin2, -1, kernel)

    return 0.5 * np.arctan2(sin2, cos2)