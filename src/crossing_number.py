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


def prune_skeleton(binary, min_branch_length=10):

    work = binary.copy()
    rows, cols = work.shape

    for _ in range(min_branch_length):
        to_remove = []
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                if work[x, y] == 1:
                    neighbors = get_neighbors(work, x, y)
                    cn = crossing_number(neighbors)
                    if cn == 1:
                        to_remove.append((x, y))
        if not to_remove:
            break
        for x, y in to_remove:
            work[x, y] = 0

    return work


def remove_isolated_bifurcations(binary, bifurcations, min_branch_length=8):
    if not bifurcations:
        return []

    rows, cols = binary.shape
    result = []

    for bx, by in bifurcations:
        valid = True
        neighbors_coords = [
            (bx - 1, by), (bx - 1, by + 1), (bx, by + 1), (bx + 1, by + 1),
            (bx + 1, by), (bx + 1, by - 1), (bx, by - 1), (bx - 1, by - 1),
        ]

        branch_lengths = []
        for nx, ny in neighbors_coords:
            if 0 <= nx < rows and 0 <= ny < cols and binary[nx, ny] == 1:
                length = trace_branch(binary, nx, ny, bx, by, max_len=min_branch_length + 2)
                branch_lengths.append(length)

        if branch_lengths and min(branch_lengths) < min_branch_length:
            valid = False

        if valid:
            result.append((bx, by))

    return result


def trace_branch(binary, start_x, start_y, skip_x, skip_y, max_len=10):
    rows, cols = binary.shape
    visited = {(skip_x, skip_y), (start_x, start_y)}
    cur_x, cur_y = start_x, start_y
    length = 1

    while length < max_len:
        next_px = None
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cur_x + dx, cur_y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if binary[nx, ny] == 1 and (nx, ny) not in visited:
                        next_px = (nx, ny)
                        break
            if next_px:
                break

        if next_px is None:
            break

        visited.add(next_px)
        cur_x, cur_y = next_px
        length += 1

    return length


def restore_endings(pruned, original, max_distance=12):
    return pruned


def remove_border_points_by_mask(points, roi_mask, margin=12):
    if not points:
        return []

    rows, cols = roi_mask.shape
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
    safe_mask = cv.erode(roi_mask.astype(np.uint8), kernel)

    filtered = []
    for p in points:
        x, y = p[0], p[1]
        if 0 <= x < rows and 0 <= y < cols and safe_mask[x, y] > 0:
            filtered.append(p)
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


def remove_paired_endings(endings, min_distance=15):
    if not endings:
        return []

    arr = np.array([(e[0], e[1]) for e in endings])
    keep = np.ones(len(endings), dtype=bool)

    for i in range(len(endings)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(endings)):
            if not keep[j]:
                continue
            d = np.sqrt((arr[i, 0] - arr[j, 0]) ** 2 + (arr[i, 1] - arr[j, 1]) ** 2)
            if d < min_distance:
                keep[i] = False
                keep[j] = False

    return [endings[i] for i in range(len(endings)) if keep[i]]


def remove_ending_near_bifurcation(endings, bifurcations, min_distance=15):
    if not endings or not bifurcations:
        return endings

    result = []
    for e in endings:
        too_close = False
        for b in bifurcations:
            d = np.sqrt((e[0] - b[0]) ** 2 + (e[1] - b[1]) ** 2)
            if d < min_distance:
                too_close = True
                break
        if not too_close:
            result.append(e)
    return result


def compute_minutia_orientation(binary, x, y, window=7):
    rows, cols = binary.shape
    x0 = max(0, x - window)
    x1 = min(rows, x + window + 1)
    y0 = max(0, y - window)
    y1 = min(cols, y + window + 1)

    block = binary[x0:x1, y0:y1]
    ridge_points = np.argwhere(block == 1)

    if len(ridge_points) < 2:
        return 0.0

    cov = np.cov(ridge_points.T)
    if cov.ndim == 0:
        return 0.0

    try:
        _, eigenvectors = np.linalg.eigh(cov)
        main_vec = eigenvectors[:, -1]
        return float(np.arctan2(main_vec[0], main_vec[1]))
    except np.linalg.LinAlgError:
        return 0.0


def extract_minutiae_points(image, roi_mask=None):
    binary = np.zeros_like(image, dtype=np.uint8)
    binary[image < 128] = 1

    pruned = prune_skeleton(binary, min_branch_length=10)

    endings = []
    bifurcations = []

    rows, cols = pruned.shape
    for x in range(1, rows - 1):
        for y in range(1, cols - 1):
            if pruned[x, y] == 1:
                neighbors = get_neighbors(pruned, x, y)
                cn = crossing_number(neighbors)
                if cn == 1:
                    endings.append((x, y))
                elif cn == 3:
                    bifurcations.append((x, y))

    if roi_mask is not None:
        endings = remove_border_points_by_mask(endings, roi_mask, margin=14)
        bifurcations = remove_border_points_by_mask(bifurcations, roi_mask, margin=14)
    else:
        margin = 20
        endings = [e for e in endings if margin <= e[0] < rows - margin and margin <= e[1] < cols - margin]
        bifurcations = [b for b in bifurcations if margin <= b[0] < rows - margin and margin <= b[1] < cols - margin]

    endings = remove_close_points(endings, min_distance=12)
    bifurcations = remove_close_points(bifurcations, min_distance=12)

    bifurcations = remove_isolated_bifurcations(pruned, bifurcations, min_branch_length=8)

    endings = remove_paired_endings(endings, min_distance=15)

    endings = remove_ending_near_bifurcation(endings, bifurcations, min_distance=15)

    endings_full = [(x, y, compute_minutia_orientation(pruned, x, y), 'ending')
                    for x, y in endings]
    bifurcations_full = [(x, y, compute_minutia_orientation(pruned, x, y), 'bifurcation')
                         for x, y in bifurcations]

    return endings_full, bifurcations_full


def draw_minutiae(image, endings, bifurcations):
    if len(image.shape) == 2:
        result = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    else:
        result = image.copy()

    for m in endings:
        x, y = m[0], m[1]
        cv.circle(result, (y, x), 4, (0, 255, 0), 1)
        if len(m) >= 3:
            theta = m[2]
            dx = int(10 * np.cos(theta))
            dy = int(10 * np.sin(theta))
            cv.line(result, (y, x), (y + dy, x + dx), (0, 255, 0), 1)

    for m in bifurcations:
        x, y = m[0], m[1]
        cv.circle(result, (y, x), 4, (0, 0, 255), 1)
        if len(m) >= 3:
            theta = m[2]
            dx = int(10 * np.cos(theta))
            dy = int(10 * np.sin(theta))
            cv.line(result, (y, x), (y + dy, x + dx), (0, 0, 255), 1)

    return result