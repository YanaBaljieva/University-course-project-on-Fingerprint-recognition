import numpy as np
import cv2 as cv


def normalise_zero_mean_unit_variance(img):
    img = img.astype(np.float32)
    std = np.std(img)
    if std < 1e-8:
        return img - np.mean(img)
    return (img - np.mean(img)) / std


def create_segmented_and_variance_images(im, block_size, threshold=0.2):
    
    im = im.astype(np.float32)
    rows, cols = im.shape

    image_variance = np.zeros_like(im, dtype=np.float32)

    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            block = im[r:min(r + block_size, rows), c:min(c + block_size, cols)]
            stddev = np.std(block)
            image_variance[r:min(r + block_size, rows), c:min(c + block_size, cols)] = stddev

    global_std = np.std(im)
    thresh = global_std * threshold

    mask = np.ones_like(im, dtype=np.uint8)
    mask[image_variance < thresh] = 0

    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (block_size * 2, block_size * 2)
    )
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    segmented_image = im * mask
    norm_img = normalise_zero_mean_unit_variance(im)

    roi = norm_img[mask > 0]
    if roi.size > 0:
        roi_mean = np.mean(roi)
        roi_std = np.std(roi)
        if roi_std < 1e-8:
            roi_std = 1.0
        norm_img = (norm_img - roi_mean) / roi_std

    norm_img = norm_img * mask

    return segmented_image.astype(np.float32), norm_img.astype(np.float32), mask.astype(np.uint8)