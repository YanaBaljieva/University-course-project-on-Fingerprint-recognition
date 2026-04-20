import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from normalization import normalize
from segmentation import create_segmented_and_variance_images
from orientation import calculate_angles
from frequency import ridge_freq
from gabor_filter import gabor_filter
from skeletonize import skeletonize
from crossing_number import extract_minutiae_points, draw_minutiae
from matching import match_minutiae
from evaluation import compute_all_pairs, compute_roc, summarize, get_class_id


DATASET_PATH = "../fingerprints/DB1_B"
BLOCK_SIZE = 16
OUTPUT_DIR = "./results"


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
        normim, mask, angles, BLOCK_SIZE,
        kernel_size=5, minWaveLength=5, maxWaveLength=15,
    )
    gabor_img = gabor_filter(normim, angles, freq, block_size=BLOCK_SIZE)
    thin_img = skeletonize(gabor_img)

    endings, bifurcations = extract_minutiae_points(thin_img, roi_mask=mask)
    minutiae_img = draw_minutiae(thin_img, endings, bifurcations)

    return {
        "original": img,
        "normalized": normalized,
        "segmented": segmented,
        "mask": mask,
        "gabor": gabor_img,
        "skeleton": thin_img,
        "endings": endings,
        "bifurcations": bifurcations,
        "all_minutiae": endings + bifurcations,
        "minutiae_img": minutiae_img,
    }


def list_images(dataset_path):
    exts = ('.tif', '.tiff', '.png', '.jpg', '.bmp')
    files = []
    for f in sorted(os.listdir(dataset_path)):
        if f.lower().endswith(exts):
            files.append(os.path.join(dataset_path, f))
    return files


def show_pipeline(result, title_prefix="Fingerprint", save_path=None):
    images = [
        result["original"],
        result["segmented"],
        result["gabor"],
        result["skeleton"],
        result["minutiae_img"],
    ]
    titles = [
        "1. Original",
        "2. Segmented",
        "3. Gabor Enhanced",
        "4. Skeleton",
        "5. Minutiae (green=ending, red=bifurcation)",
    ]

    plt.figure(figsize=(15, 8))
    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(2, 3, i)
        if len(img.shape) == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")

    plt.suptitle(title_prefix, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def show_score_distributions(genuine, impostor, save_path=None):
    plt.figure(figsize=(10, 5))
    bins = np.linspace(0, 1, 30)

    plt.hist(impostor, bins=bins, alpha=0.6,
             label=f'Impostor (N={len(impostor)})',
             color='red', edgecolor='darkred')
    plt.hist(genuine, bins=bins, alpha=0.6,
             label=f'Genuine (N={len(genuine)})',
             color='green', edgecolor='darkgreen')

    if genuine:
        plt.axvline(np.mean(genuine), color='green', linestyle='--',
                    label=f'Genuine mean = {np.mean(genuine):.3f}')
    if impostor:
        plt.axvline(np.mean(impostor), color='red', linestyle='--',
                    label=f'Impostor mean = {np.mean(impostor):.3f}')

    plt.xlabel('Matching score')
    plt.ylabel('Frequency')
    plt.title('Genuine vs Impostor score distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def show_roc(roc, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(roc['far'], roc['frr'], linewidth=2, color='blue', label='ROC')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    plt.scatter([roc['eer']], [roc['eer']], color='red', s=120, zorder=5,
                label=f"EER = {roc['eer']:.3f}")

    plt.xlabel('False Accept Rate (FAR)')
    plt.ylabel('False Reject Rate (FRR)')
    plt.title(f"ROC curve (AUC = {roc['auc']:.3f})")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def show_matching(img1, img2, name1, name2, score, title, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title(name1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title(name2)
    plt.axis("off")

    plt.suptitle(f"{title}  -  score: {score:.4f}", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def find_representative_pairs(records):

    same = [r for r in records if r['same']]
    diff = [r for r in records if not r['same']]

    if not same or not diff:
        return None

    same_sorted = sorted(same, key=lambda r: r['score'], reverse=True)
    diff_sorted = sorted(diff, key=lambda r: r['score'], reverse=True)

    return {
        'best_genuine': same_sorted[0],          
        'worst_genuine': same_sorted[-1],       
        'worst_impostor': diff_sorted[0],      
        'best_impostor': diff_sorted[-1],      
    }


def save_pair_image(record, all_results, dataset_path, title, output_path):
    f1 = os.path.join(dataset_path, record['file1'])
    f2 = os.path.join(dataset_path, record['file2'])
    show_matching(
        all_results[f1]["minutiae_img"],
        all_results[f2]["minutiae_img"],
        record['file1'], record['file2'],
        record['score'], title,
        save_path=output_path,
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Система за разпознаване на пръстови отпечатъци")
    print("=" * 60)

    files = list_images(DATASET_PATH)
    print(f"\nDataset: {DATASET_PATH}")
    print(f"Намерени: {len(files)} изображения")
    if len(files) == 0:
        print("ГРЕШКА: няма намерени изображения.")
        return

    print(f"\n[1] Обработка на всички изображения...")
    t0 = time.time()
    features_by_file = {}
    all_results = {}

    for idx, path in enumerate(files, start=1):
        name = os.path.basename(path)
        try:
            result = process_image(path)
            features_by_file[path] = result["all_minutiae"]
            all_results[path] = result
            n_end = len(result["endings"])
            n_bif = len(result["bifurcations"])
            print(f"  {idx:3d}/{len(files)} {name:15s} -> "
                  f"{n_end:3d} endings + {n_bif:3d} bifurcations "
                  f"= {n_end + n_bif} minutiae")
        except Exception as e:
            print(f"  {idx:3d}/{len(files)} {name:15s} -> ГРЕШКА: {e}")

    print(f"Време за обработка: {time.time() - t0:.1f} сек")

    print(f"\n[2] Визуализация на pipeline...")
    first_path = list(all_results.keys())[0]
    show_pipeline(
        all_results[first_path],
        title_prefix=f"Pipeline: {os.path.basename(first_path)}",
        save_path=os.path.join(OUTPUT_DIR, "pipeline_example.png"),
    )
    print(f"  Запазено: {OUTPUT_DIR}/pipeline_example.png")

    print(f"\n[3] Изчисляване на matching scores за всички двойки...")
    t0 = time.time()
    n_pairs = len(features_by_file) * (len(features_by_file) - 1) // 2
    print(f"  Общо двойки: {n_pairs}")

    genuine, impostor, records = compute_all_pairs(features_by_file, match_minutiae)

    print(f"  Време: {time.time() - t0:.1f} сек")
    print(f"  Genuine:  {len(genuine)} двойки")
    print(f"  Impostor: {len(impostor)} двойки")

    print(f"\n[4] Изчисляване на ROC и EER...")
    roc = compute_roc(genuine, impostor)

    print(f"\n[5] Визуализации...")
    show_score_distributions(
        genuine, impostor,
        save_path=os.path.join(OUTPUT_DIR, "score_distributions.png"),
    )
    show_roc(roc, save_path=os.path.join(OUTPUT_DIR, "roc_curve.png"))
    print(f"  Запазено: score_distributions.png, roc_curve.png")

    print(f"\n[6] Намиране на представителни двойки за документацията...")
    rep = find_representative_pairs(records)

    if rep is None:
        print("  Не е възможно да се намерят представителни двойки.")
        return

    bg = rep['best_genuine']
    wg = rep['worst_genuine']
    wi = rep['worst_impostor']
    bi = rep['best_impostor']

    print(f"  Best genuine  (SAME, най-висок score):  "
          f"{bg['file1']} vs {bg['file2']} -> {bg['score']:.4f}")
    print(f"  Worst genuine (SAME, най-нисък score):  "
          f"{wg['file1']} vs {wg['file2']} -> {wg['score']:.4f}")
    print(f"  Worst impostor (DIFF, най-висок score): "
          f"{wi['file1']} vs {wi['file2']} -> {wi['score']:.4f}")
    print(f"  Best impostor (DIFF, най-нисък score):  "
          f"{bi['file1']} vs {bi['file2']} -> {bi['score']:.4f}")

    save_pair_image(
        bg, all_results, DATASET_PATH,
        "Best genuine match (same finger - highest score)",
        os.path.join(OUTPUT_DIR, "match_best_genuine.png"),
    )
    save_pair_image(
        wg, all_results, DATASET_PATH,
        "Worst genuine match (same finger - lowest score)",
        os.path.join(OUTPUT_DIR, "match_worst_genuine.png"),
    )
    save_pair_image(
        wi, all_results, DATASET_PATH,
        "Worst impostor (different fingers - highest score)",
        os.path.join(OUTPUT_DIR, "match_worst_impostor.png"),
    )
    save_pair_image(
        bi, all_results, DATASET_PATH,
        "Best impostor (different fingers - lowest score)",
        os.path.join(OUTPUT_DIR, "match_best_impostor.png"),
    )

    print(f"  Запазени: match_best_genuine.png, match_worst_genuine.png,")
    print(f"            match_worst_impostor.png, match_best_impostor.png")

    save_pair_image(
        bg, all_results, DATASET_PATH,
        "Same finger", os.path.join(OUTPUT_DIR, "match_same.png"),
    )
    save_pair_image(
        bi, all_results, DATASET_PATH,
        "Different fingers", os.path.join(OUTPUT_DIR, "match_diff.png"),
    )

    print()
    print(summarize(genuine, impostor, roc))

    # Топ списъци
    same_records = sorted([r for r in records if r['same']],
                          key=lambda r: r['score'], reverse=True)
    diff_records = sorted([r for r in records if not r['same']],
                          key=lambda r: r['score'], reverse=True)

    print("\nТоп 5 SAME двойки (най-високи scores):")
    for r in same_records[:5]:
        print(f"  {r['file1']:15s} vs {r['file2']:15s}  score = {r['score']:.4f}")

    print("\nТоп 5 DIFF двойки (най-високи scores - потенциални false accepts):")
    for r in diff_records[:5]:
        print(f"  {r['file1']:15s} vs {r['file2']:15s}  score = {r['score']:.4f}")

    csv_path = os.path.join(OUTPUT_DIR, "all_pairs.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("file1,file2,same,score\n")
        for r in records:
            f.write(f"{r['file1']},{r['file2']},{int(r['same'])},{r['score']:.6f}\n")
    print(f"\nВсички двойки: {csv_path}")


if __name__ == "__main__":
    main()