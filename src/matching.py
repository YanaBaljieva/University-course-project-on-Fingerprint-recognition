import numpy as np


def center_points(points):
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    pts = np.array(points, dtype=np.float32)
    centroid = np.mean(pts, axis=0)
    return pts - centroid


def match_minutiae(minutiae1, minutiae2, threshold=10.0):
    
    if len(minutiae1) == 0 or len(minutiae2) == 0:
        return 0.0

    pts1 = center_points(minutiae1)
    pts2 = center_points(minutiae2)

    matched = 0
    used = set()

    for p1 in pts1:
        best_dist = float("inf")
        best_idx = -1

        for idx, p2 in enumerate(pts2):
            if idx in used:
                continue

            dist = np.linalg.norm(p1 - p2)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx != -1 and best_dist <= threshold:
            matched += 1
            used.add(best_idx)

    return matched / max(len(pts1), len(pts2), 1)