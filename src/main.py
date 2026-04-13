import os
import cv2
import matplotlib.pyplot as plt

from normalization import normalize
from segmentation import create_segmented_and_variance_images
from orientation import calculate_angles
from frequency import ridge_freq
from gabor_filter import gabor_filter
from skeletonize import skeletonize
from crossing_number import extract_minutiae_points, draw_minutiae
from matching import match_minutiae


DATASET_PATH = "../fingerprints/DB1_B"
BLOCK_SIZE = 16


def process_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")

    normalized = normalize(img, 100, 100)

    segmented, normim, mask = create_segmented_and_variance_images(
        normalized, BLOCK_SIZE, 0.2
    )

    angles = calculate_angles(normalized, W=BLOCK_SIZE, smooth=True)

    freq = ridge_freq(
        normim,
        mask,
        angles,
        BLOCK_SIZE,
        kernel_size=5,
        minWaveLength=5,
        maxWaveLength=15,
    )

    gabor_img = gabor_filter(normim, angles, freq, block_size=BLOCK_SIZE)
    thin_img = skeletonize(gabor_img)

    endings, bifurcations = extract_minutiae_points(thin_img)
    minutiae_img = draw_minutiae(thin_img, endings, bifurcations)

    return {
        "original": img,
        "normalized": normalized,
        "segmented": segmented,
        "gabor": gabor_img,
        "skeleton": thin_img,
        "endings": endings,
        "bifurcations": bifurcations,
        "minutiae_img": minutiae_img,
    }


def show_pipeline(result, title_prefix="Fingerprint"):
    images = [
        result["original"],
        result["segmented"],
        result["gabor"],
        result["skeleton"],
        result["minutiae_img"],
    ]
    titles = [
        f"{title_prefix} - Original",
        f"{title_prefix} - Segmented",
        f"{title_prefix} - Gabor",
        f"{title_prefix} - Skeleton",
        f"{title_prefix} - Minutiae",
    ]

    plt.figure(figsize=(14, 8))
    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(2, 3, i)
        if len(img.shape) == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()




def show_matching(result1, result2, score, title="Matching"):
    import matplotlib.pyplot as plt
    import cv2

    img1 = result1["minutiae_img"]
    img2 = result2["minutiae_img"]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("Fingerprint A")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title("Fingerprint B")
    plt.axis("off")

    plt.suptitle(f"{title} Score: {score:.4f}")
    plt.show()



def main():
    file1 = "101_1.tif"
    file2 = "101_2.tif"
    file3 = "102_1.tif"

    path1 = os.path.join(DATASET_PATH, file1)
    path2 = os.path.join(DATASET_PATH, file2)
    path3 = os.path.join(DATASET_PATH, file3)

    result1 = process_image(path1)
    result2 = process_image(path2)
    result3 = process_image(path3)

    print(file1)
    print("Endings:", len(result1["endings"]))
    print("Bifurcations:", len(result1["bifurcations"]))
    print("Total minutiae:", len(result1["endings"]) + len(result1["bifurcations"]))

    print("\n" + file2)
    print("Endings:", len(result2["endings"]))
    print("Bifurcations:", len(result2["bifurcations"]))
    print("Total minutiae:", len(result2["endings"]) + len(result2["bifurcations"]))

    print("\n" + file3)
    print("Endings:", len(result3["endings"]))
    print("Bifurcations:", len(result3["bifurcations"]))
    print("Total minutiae:", len(result3["endings"]) + len(result3["bifurcations"]))

    # Match only endings, because they are usually more stable
    score_same = match_minutiae(result1["endings"], result2["endings"], threshold=10.0)
    score_diff = match_minutiae(result1["endings"], result3["endings"], threshold=10.0)

    print("\nMatching score (same finger):", round(score_same, 4))
    print("Matching score (different finger):", round(score_diff, 4))

    show_pipeline(result1, title_prefix=file1)
    show_pipeline(result2, title_prefix=file2)

    show_matching(result1, result2, score_same, "Same finger")
    show_matching(result1, result3, score_diff, "Different finger")


if __name__ == "__main__":
    main()