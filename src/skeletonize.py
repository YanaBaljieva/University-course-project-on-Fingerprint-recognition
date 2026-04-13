import numpy as np
import cv2 as cv
from skimage.morphology import skeletonize as sk_skeletonize


def skeletonize(image_input):
    
    image = np.zeros_like(image_input, dtype=np.uint8)
    image[image_input == 0] = 1

    skeleton = sk_skeletonize(image > 0)

    output = np.zeros_like(image_input, dtype=np.uint8)
    output[skeleton] = 255
    cv.bitwise_not(output, output)

    return output