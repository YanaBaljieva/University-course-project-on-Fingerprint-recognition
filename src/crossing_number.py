import cv2 as cv
import numpy as np


def get_neighbors(img, x, y):
    return [
        img[x - 1, y],
        img[x - 1, y + 1],
        img[x, y + 1],
        img[x + 1, y + 1],
        img[x + 1, y],
        img[x + 1, y - 1],
        img[x, y - 1],
        img[x - 1, y - 1],
    ]


def crossing_number(neighbors):
    cn = 0
    for i in range(8):
        cn += abs(int(neighbors[i]) - int(neighbors[(i + 1) % 8]))
    return cn // 2


def remove_border_points(points, shape, margin=15):
    rows, cols = shape
    filtered = []
    for x, y in points:
        if margin <= x < rows - margin and margin <= y < cols - margin:
            filtered.append((x, y))
    return filtered


def remove_close_points(points, min_distance=10):
    if not points:
        return []

    filtered = []
    for p in points:
        keep = True
        for q in filtered:
            dist = np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
            if dist < min_distance:
                keep = False
                break
        if keep:
            filtered.append(p)
    return filtered


def extract_minutiae_points(image):
    
    binary = np.zeros_like(image, dtype=np.uint8)
    binary[image < 10] = 1

    endings = []
    bifurcations = []

    rows, cols = binary.shape

    for x in range(1, rows - 1):
        for y in range(1, cols - 1):
            if binary[x, y] == 1:
                neighbors = get_neighbors(binary, x, y)
                cn = crossing_number(neighbors)

                if cn == 1:
                    endings.append((x, y))
                elif cn == 3:
                    bifurcations.append((x, y))

    endings = remove_border_points(endings, binary.shape, margin=15)
    bifurcations = remove_border_points(bifurcations, binary.shape, margin=15)

    endings = remove_close_points(endings, min_distance=10)
    bifurcations = remove_close_points(bifurcations, min_distance=10)

    return endings, bifurcations


def draw_minutiae(image, endings, bifurcations):
    
    if len(image.shape) == 2:
        result = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    else:
        result = image.copy()

    for x, y in endings:
        cv.circle(result, (y, x), 3, (0, 255, 0), 1)

    for x, y in bifurcations:
        cv.circle(result, (y, x), 3, (0, 0, 255), 1)

    return result