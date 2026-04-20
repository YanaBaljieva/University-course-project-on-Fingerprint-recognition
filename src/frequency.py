import numpy as np
import math
import scipy.ndimage


def frequest(im, orient_block, kernel_size, min_wave_length, max_wave_length):
    rows, cols = im.shape

    cosorient = np.cos(2 * orient_block)
    sinorient = np.sin(2 * orient_block)
    block_orient = math.atan2(np.mean(sinorient), np.mean(cosorient)) / 2

    rotim = scipy.ndimage.rotate(
        im,
        block_orient / np.pi * 180 + 90,
        axes=(1, 0),
        reshape=False,
        order=3,
        mode="nearest",
    )

    cropsze = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - cropsze) / 2))
    rotim = rotim[offset:offset + cropsze, offset:offset + cropsze]

    ridge_sum = np.sum(rotim, axis=0)
    dilation = scipy.ndimage.grey_dilation(
        ridge_sum, size=kernel_size, structure=np.ones(kernel_size)
    )
    ridge_noise = np.abs(dilation - ridge_sum)
    peak_thresh = 2

    maxpts = (ridge_noise < peak_thresh) & (ridge_sum > np.mean(ridge_sum))
    maxind = np.where(maxpts)[0]
    no_of_peaks = len(maxind)

    if no_of_peaks < 2:
        return np.zeros(im.shape, dtype=np.float32)

    wave_length = (maxind[-1] - maxind[0]) / (no_of_peaks - 1)

    if min_wave_length <= wave_length <= max_wave_length:
        return (1.0 / wave_length) * np.ones(im.shape, dtype=np.float32)

    return np.zeros(im.shape, dtype=np.float32)


def ridge_freq(im, mask, orient, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15):
    rows, cols = im.shape
    freq = np.zeros((rows, cols), dtype=np.float32)

    orient_rows, orient_cols = orient.shape

    for r in range(orient_rows):
        for c in range(orient_cols):
            r0 = r * block_size
            c0 = c * block_size

            if r0 + block_size > rows or c0 + block_size > cols:
                continue

            block_mask = mask[r0:r0 + block_size, c0:c0 + block_size]
            if np.mean(block_mask) < 0.5:
                continue

            image_block = im[r0:r0 + block_size, c0:c0 + block_size]
            angle_block = orient[r, c]

            freq[r0:r0 + block_size, c0:c0 + block_size] = frequest(
                image_block,
                angle_block,
                kernel_size,
                minWaveLength,
                maxWaveLength,
            )

    freq = freq * mask

    non_zero = freq[freq > 0]
    if non_zero.size == 0:
        return freq

    medianfreq = np.median(non_zero)
    return medianfreq * mask