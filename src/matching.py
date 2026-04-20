import numpy as np


def minutiae_to_array(minutiae):
    if len(minutiae) == 0:
        return np.empty((0, 4), dtype=np.float32)

    arr = np.zeros((len(minutiae), 4), dtype=np.float32)
    for i, m in enumerate(minutiae):
        arr[i, 0] = m[0]
        arr[i, 1] = m[1]
        arr[i, 2] = m[2] if len(m) >= 3 else 0.0
        if len(m) >= 4:
            arr[i, 3] = 0.0 if m[3] == 'ending' else 1.0
    return arr


def rotate_points(arr, angle_rad, center):
    if len(arr) == 0:
        return arr.copy()

    rotated = arr.copy()
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    dx = arr[:, 0] - center[0]
    dy = arr[:, 1] - center[1]

    rotated[:, 0] = cos_a * dx - sin_a * dy + center[0]
    rotated[:, 1] = sin_a * dx + cos_a * dy + center[1]
    rotated[:, 2] = arr[:, 2] + angle_rad

    return rotated


def angle_diff(a1, a2):
    diff = np.abs(a1 - a2) % np.pi
    return np.minimum(diff, np.pi - diff)


def count_matches(pts1, pts2, dist_threshold=15.0, angle_threshold=np.pi / 6):

    n1, n2 = len(pts1), len(pts2)
    if n1 == 0 or n2 == 0:
        return 0

    dx = pts1[:, 0:1] - pts2[:, 0:1].T
    dy = pts1[:, 1:2] - pts2[:, 1:2].T
    dist = np.sqrt(dx * dx + dy * dy)

    ang = angle_diff(pts1[:, 2:3], pts2[:, 2:3].T)

    type_ok = pts1[:, 3:4] == pts2[:, 3:4].T

    valid = (dist <= dist_threshold) & (ang <= angle_threshold) & type_ok

    row_idx, col_idx = np.where(valid)
    if len(row_idx) == 0:
        return 0

    dists = dist[row_idx, col_idx]
    order = np.argsort(dists)

    used_r = np.zeros(n1, dtype=bool)
    used_c = np.zeros(n2, dtype=bool)
    matched = 0

    for k in order:
        i, j = row_idx[k], col_idx[k]
        if not used_r[i] and not used_c[j]:
            used_r[i] = True
            used_c[j] = True
            matched += 1

    return matched


def match_minutiae(minutiae1, minutiae2,
                   dist_threshold=15.0,
                   angle_threshold=np.pi / 6):

    pts1 = minutiae_to_array(minutiae1)
    pts2 = minutiae_to_array(minutiae2)

    if len(pts1) == 0 or len(pts2) == 0:
        return 0.0

    min_count = min(len(pts1), len(pts2))
    if min_count == 0:
        return 0.0

    center1 = np.mean(pts1[:, :2], axis=0)
    center2 = np.mean(pts2[:, :2], axis=0)

    best_score = 0.0

    for angle_deg in range(-30, 31, 5):
        angle_rad = np.radians(angle_deg)
        rotated = rotate_points(pts2, angle_rad, center=center2)

        translation = center1 - center2
        aligned = rotated.copy()
        aligned[:, 0] += translation[0]
        aligned[:, 1] += translation[1]

        matched = count_matches(
            pts1, aligned,
            dist_threshold=dist_threshold,
            angle_threshold=angle_threshold,
        )
        score = matched / min_count
        if score > best_score:
            best_score = score

    return best_score