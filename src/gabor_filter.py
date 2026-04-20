import numpy as np
import scipy.ndimage


def gabor_filter(im, orient, freq, kx=0.65, ky=0.65, block_size=16):
    angle_inc = 3
    im = np.asarray(im, dtype=np.float32)
    rows, cols = im.shape
    return_img = np.zeros((rows, cols), dtype=np.float32)

    freq_1d = freq.flatten()
    non_zero = freq_1d[freq_1d > 0]

    if non_zero.size == 0:
        out = np.zeros_like(im, dtype=np.uint8)
        out[im < np.mean(im)] = 255
        return out

    non_zero = np.round(non_zero * 100) / 100
    unfreq = np.unique(non_zero)

    sigma_x = (1 / unfreq) * kx
    sigma_y = (1 / unfreq) * ky
    filt_size = int(np.round(3 * np.max([np.max(sigma_x), np.max(sigma_y)])))
    if filt_size < 1:
        filt_size = 1

    arr = np.linspace(-filt_size, filt_size, (2 * filt_size + 1))
    x, y = np.meshgrid(arr, arr)

    reffilter = np.exp(
        -(((x ** 2) / (sigma_x[0] ** 2)) + ((y ** 2) / (sigma_y[0] ** 2)))
    ) * np.cos(2 * np.pi * unfreq[0] * x)

    filt_rows, filt_cols = reffilter.shape
    gabor_bank = np.zeros((180 // angle_inc, filt_rows, filt_cols), dtype=np.float32)

    for degree in range(180 // angle_inc):
        rot_filt = scipy.ndimage.rotate(
            reffilter, -(degree * angle_inc + 90), reshape=False
        )
        gabor_bank[degree] = rot_filt

    max_orient_index = int(np.round(180 / angle_inc))
    orientindex = np.round(orient / np.pi * 180 / angle_inc).astype(int)

    for r in range(orientindex.shape[0]):
        for c in range(orientindex.shape[1]):
            if orientindex[r, c] < 1:
                orientindex[r, c] += max_orient_index
            if orientindex[r, c] > max_orient_index:
                orientindex[r, c] -= max_orient_index

    valid_row, valid_col = np.where(freq > 0)
    valid = np.where(
        (valid_row > filt_size)
        & (valid_row < rows - filt_size)
        & (valid_col > filt_size)
        & (valid_col < cols - filt_size)
    )[0]

    for k in valid:
        r = valid_row[k]
        c = valid_col[k]
        img_block = im[r - filt_size:r + filt_size + 1, c - filt_size:c + filt_size + 1]

        br = min(r // block_size, orientindex.shape[0] - 1)
        bc = min(c // block_size, orientindex.shape[1] - 1)
        filt_idx = orientindex[br, bc] - 1

        return_img[r, c] = np.sum(img_block * gabor_bank[filt_idx])

    gabor_img = 255 - ((return_img < 0).astype(np.uint8) * 255)
    return gabor_img