import os
import re
import numpy as np


def get_class_id(filename):
    base = os.path.basename(filename)
    match = re.match(r'(\d+)_\d+', base)
    if match:
        return match.group(1)
    return base.split('_')[0]


def compute_all_pairs(features_by_file, match_fn):
    files = sorted(features_by_file.keys())
    genuine = []
    impostor = []
    records = []

    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            f1, f2 = files[i], files[j]
            c1, c2 = get_class_id(f1), get_class_id(f2)

            score = match_fn(
                features_by_file[f1],
                features_by_file[f2],
            )

            same = (c1 == c2)
            if same:
                genuine.append(score)
            else:
                impostor.append(score)

            records.append({
                'file1': os.path.basename(f1),
                'file2': os.path.basename(f2),
                'same': same,
                'score': score,
            })

    return genuine, impostor, records


def compute_roc(genuine, impostor, num_thresholds=200):
    if len(genuine) == 0 or len(impostor) == 0:
        return {
            'thresholds': [],
            'far': [],
            'frr': [],
            'eer': 0.5,
            'eer_threshold': 0.5,
            'auc': 0.5,
        }

    gen = np.array(genuine)
    imp = np.array(impostor)

    lo = min(gen.min(), imp.min())
    hi = max(gen.max(), imp.max())
    thresholds = np.linspace(lo, hi, num_thresholds)

    far_list = np.zeros(num_thresholds)
    frr_list = np.zeros(num_thresholds)

    for k, t in enumerate(thresholds):
        far_list[k] = np.mean(imp >= t)
        frr_list[k] = np.mean(gen < t)

    diff = np.abs(far_list - frr_list)
    eer_idx = np.argmin(diff)
    eer = (far_list[eer_idx] + frr_list[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    tpr = 1.0 - frr_list 
    sort_idx = np.argsort(far_list)
    far_sorted = far_list[sort_idx]
    tpr_sorted = tpr[sort_idx]
    auc = np.trapz(tpr_sorted, far_sorted)

    return {
        'thresholds': thresholds.tolist(),
        'far': far_list.tolist(),
        'frr': frr_list.tolist(),
        'eer': float(eer),
        'eer_threshold': float(eer_threshold),
        'auc': float(auc),
    }


def summarize(genuine, impostor, roc):
    lines = []
    lines.append("=" * 60)
    lines.append("ОБОБЩЕНИЕ НА РЕЗУЛТАТИТЕ")
    lines.append("=" * 60)
    lines.append(f"Брой genuine двойки:   {len(genuine)}")
    lines.append(f"Брой impostor двойки:  {len(impostor)}")

    if genuine:
        lines.append(f"Genuine score  mean={np.mean(genuine):.4f} "
                     f"std={np.std(genuine):.4f} "
                     f"min={np.min(genuine):.4f} max={np.max(genuine):.4f}")
    if impostor:
        lines.append(f"Impostor score mean={np.mean(impostor):.4f} "
                     f"std={np.std(impostor):.4f} "
                     f"min={np.min(impostor):.4f} max={np.max(impostor):.4f}")

    if genuine and impostor:
        gap = np.mean(genuine) - np.mean(impostor)
        lines.append(f"Separation (genuine mean - impostor mean): {gap:.4f}")

    lines.append("")
    lines.append(f"EER:            {roc['eer']:.4f}")
    lines.append(f"EER threshold:  {roc['eer_threshold']:.4f}")
    lines.append(f"AUC:            {roc['auc']:.4f}")
    lines.append("=" * 60)

    return "\n".join(lines)