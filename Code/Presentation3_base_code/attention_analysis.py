"""
Attention analysis and visualization utilities.

Presentation 2, Analyst choice 1 changes:
  * Each attention score is paired with its patch BY NAME. The dataset returns
    "stain_paths" aligned row-for-row with "stain_slices", so patch_weights[s][i]
    belongs to stain_paths[stain][s][i]. The old approach (re-globbing the
    folder and sorting patch numbers, which disagreed with the alphabetical
    order used in data preparation) is removed.
  * No fallbacks: a missing or unreadable patch image, or attention weights
    whose shape does not match the patches, raises an error.
  * Every patch record and CSV row carries its patch filename and full path.
"""

import os
import re
import csv
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch


# ---------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------
def analyze_attention_weights(model, test_loader, output_dir: str, top_n: int = 5):
    print("\n" + "=" * 60)
    print("ATTENTION ANALYSIS")
    print("=" * 60)

    attention_dir = os.path.join(output_dir, "attention_analysis")
    os.makedirs(attention_dir, exist_ok=True)
    plots_dir = os.path.join(attention_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    model.eval()
    attention_summary = []
    all_patch_records = []
    case_label_info = {}
    slice_attention_records = []

    with torch.no_grad():
        for batch in test_loader:
            case_data = batch[0]
            case_id = case_data["case_id"]
            stain_slices = case_data["stain_slices"]
            if "stain_paths" not in case_data:
                raise KeyError("Dataset did not return 'stain_paths'; use the updated dataset.py")
            stain_paths = case_data["stain_paths"]

            logits, attention_weights = model(stain_slices, return_attn_weights=True)
            check_alignment(case_id, attention_weights, stain_slices, stain_paths)

            true_label = int(case_data["label"].detach().view(-1)[0].item())
            pred_label = int(torch.argmax(logits.detach().view(1, -1), dim=1).item())
            case_label_info[case_id] = {"true_label": true_label, "pred_label": pred_label}

            attention_summary.append(
                analyze_case_attention(case_id, stain_paths, attention_weights, attention_dir, top_n)
            )

            patch_records = compute_effective_patch_attention(case_id, attention_weights, stain_paths)
            all_patch_records.extend(patch_records)

            visualize_case_effective_patches(case_id, patch_records, attention_dir, top_n)

            for stain, weights_dict in attention_weights["stain_weights"].items():
                sw = weights_dict["slice_weights"].detach().cpu().numpy().reshape(-1)
                for slice_idx, w in enumerate(sw):
                    slice_attention_records.append({
                        "case_id": case_id,
                        "stain": stain,
                        "slice_idx": int(slice_idx),
                        "slice_id": parse_slice_id(stain_paths[stain][slice_idx][0]),
                        "slice_attn_weight": float(w),
                    })

    save_attention_summary(attention_summary, attention_dir)
    save_all_patch_records(all_patch_records, case_label_info, attention_dir)
    plot_effective_patch_attention_distribution_per_case(all_patch_records, case_label_info, plots_dir, bins=50)
    plot_slice_attention_distribution_per_caseandstain(slice_attention_records, case_label_info, attention_dir, bins=30)
    analyze_top_effective_patches_per_case(all_patch_records, case_label_info, attention_dir, top_percent=5.0)

    print(f"Attention analysis saved to: {attention_dir}")


# ---------------------------------------------------------------------
# Name-based pairing
# ---------------------------------------------------------------------
def check_alignment(case_id, attention_weights: Dict, stain_slices: Dict, stain_paths: Dict):
    """Raise unless every attention weight has exactly one patch file, in the same order."""
    for key in ("case_weights", "stain_order", "stain_weights"):
        if key not in attention_weights:
            raise KeyError(f"Case {case_id}: attention output missing '{key}'")

    for stain, wd in attention_weights["stain_weights"].items():
        if stain not in stain_paths:
            raise KeyError(f"Case {case_id}: attention has stain '{stain}' but no patch paths for it")
        paths = stain_paths[stain]
        patch_ws = wd["patch_weights"]
        n_slice_w = wd["slice_weights"].numel()
        if not (len(paths) == len(patch_ws) == n_slice_w == len(stain_slices[stain])):
            raise ValueError(
                f"Case {case_id} {stain}: {len(paths)} slices of paths, {len(patch_ws)} patch-weight "
                f"vectors, {n_slice_w} slice weights, {len(stain_slices[stain])} slice tensors")
        for s_i, (pw, sl_paths, sl_tensor) in enumerate(zip(patch_ws, paths, stain_slices[stain])):
            if not (pw.numel() == len(sl_paths) == sl_tensor.shape[0]):
                raise ValueError(
                    f"Case {case_id} {stain} slice {s_i}: {pw.numel()} weights, "
                    f"{len(sl_paths)} paths, {sl_tensor.shape[0]} tensor rows")


def load_patch_image(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Patch image not found: {path}")
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Could not read patch image {path}") from e


def parse_slice_id(fname: str) -> str:
    """
    case_052_match_1_h&e_patch164.png   -> 'match_1'
    case_83_unmatched2_h&e_patch18.png  -> 'unmatched_2'
    """
    base = os.path.basename(fname)
    m = re.match(r"case_\d+_([a-zA-Z]+_\d+)_", base)
    if m:
        return m.group(1).lower()
    m = re.match(r"case_\d+_([a-zA-Z]+)(\d+)_", base)
    if m:
        return f"{m.group(1).lower()}_{m.group(2)}"
    raise ValueError(f"Cannot parse slice id from filename: {base}")


def parse_patch_number(fname: str) -> int:
    m = re.search(r"_patch(\d+)\.\w+$", os.path.basename(fname), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse patch number from filename: {os.path.basename(fname)}")
    return int(m.group(1))


# ---------------------------------------------------------------------
# Case-level analysis (most / least attended slice per stain)
# ---------------------------------------------------------------------
def analyze_case_attention(case_id, stain_paths: Dict, attention_weights: Dict,
                           output_dir: str, top_n: int = 5) -> Dict:
    patch_attention_dir = os.path.join(output_dir, "patch_attention")
    os.makedirs(patch_attention_dir, exist_ok=True)

    case_weights = attention_weights["case_weights"].detach().cpu().numpy().reshape(-1)
    stain_order = attention_weights["stain_order"]
    if len(case_weights) != len(stain_order):
        raise ValueError(f"Case {case_id}: {len(case_weights)} stain weights for {len(stain_order)} stains")

    summary = {
        "case_id": case_id,
        "stain_order": stain_order,
        "stain_attention": {s: float(w) for s, w in zip(stain_order, case_weights)},
        "most_attended_stain": stain_order[int(np.argmax(case_weights))] if len(stain_order) else None,
    }

    for stain, wd in attention_weights["stain_weights"].items():
        slice_w = wd["slice_weights"].detach().cpu().numpy().reshape(-1)
        if slice_w.size == 0:
            continue
        for prefix, s_idx in (("top", int(np.argmax(slice_w))), ("bottom", int(np.argmin(slice_w)))):
            visualize_patch_attention(
                case_id=case_id, stain=stain, slice_idx=s_idx,
                patch_paths=stain_paths[stain][s_idx],
                patch_weights=wd["patch_weights"][s_idx].detach().cpu().numpy().reshape(-1),
                output_dir=patch_attention_dir, top_n=top_n, prefix=prefix,
            )
    return summary


def visualize_patch_attention(case_id, stain: str, slice_idx: int, patch_paths: List[str],
                              patch_weights: np.ndarray, output_dir: str,
                              top_n: int = 5, prefix: str = "top"):
    n = len(patch_weights)
    if n == 0:
        return
    order = np.argsort(patch_weights)
    indices = order[-top_n:][::-1] if prefix == "top" else order[:top_n]
    title_prefix = "Most" if prefix == "top" else "Least"

    n_cols = min(5, len(indices))
    n_rows = (len(indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.3 * n_rows), squeeze=False)
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")

    for ax, i in zip(axes, indices):
        path = patch_paths[int(i)]
        ax.imshow(load_patch_image(path))
        ax.set_title(f"patch{parse_patch_number(path)}\nw={patch_weights[int(i)]:.4f}", fontsize=9)

    slice_id = parse_slice_id(patch_paths[0])
    fig.suptitle(f"Case {case_id} - {stain} - {slice_id}\n{title_prefix} attended patches",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = f"case_{case_id}_{stain.replace('&', 'and')}_{slice_id}_{prefix}_patches.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Effective patch attention (stain_w * slice_w * patch_w), one record per file
# ---------------------------------------------------------------------
def compute_effective_patch_attention(case_id, attention_weights: Dict,
                                      stain_paths: Dict) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    case_weights = attention_weights["case_weights"].detach().cpu().numpy().reshape(-1)

    for stain_idx, stain in enumerate(attention_weights["stain_order"]):
        wd = attention_weights["stain_weights"][stain]
        stain_w = float(case_weights[stain_idx])
        slice_w = wd["slice_weights"].detach().cpu().numpy().reshape(-1)

        for slice_idx, pw in enumerate(wd["patch_weights"]):
            pw = pw.detach().cpu().numpy().reshape(-1)
            paths = stain_paths[stain][slice_idx]
            slice_id = parse_slice_id(paths[0])
            for i, (w, path) in enumerate(zip(pw, paths)):
                records.append({
                    "case_id": case_id,
                    "stain": stain,
                    "slice_idx": int(slice_idx),
                    "slice_id": slice_id,
                    "patch_file": os.path.basename(path),
                    "patch_path": path,
                    "patch_row": i,  # position in the model's input, NOT the patch number
                    "n_patches_in_slice": len(pw),
                    "patch_attn_weight": float(w),
                    "slice_attn_weight": float(slice_w[slice_idx]),
                    "stain_attn_weight": stain_w,
                    "effective_weight": stain_w * float(slice_w[slice_idx]) * float(w),
                })
    return records


def visualize_case_effective_patches(case_id, patch_records: List[Dict[str, Any]],
                                     output_dir: str, top_n: int = 5):
    if not patch_records:
        return
    ranked = sorted(patch_records, key=lambda r: r["effective_weight"])
    k = min(top_n, len(ranked))
    out_dir = os.path.join(output_dir, "case_effective_patches")
    os.makedirs(out_dir, exist_ok=True)

    def _plot(entries, title_prefix, filename):
        n_cols = min(5, len(entries))
        n_rows = (len(entries) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.4 * n_rows), squeeze=False)
        axes = axes.flatten()
        for ax in axes:
            ax.axis("off")
        for ax, e in zip(axes, entries):
            ax.imshow(load_patch_image(e["patch_path"]))
            ax.set_title(f"{e['stain']} {e['slice_id']}\npatch{parse_patch_number(e['patch_file'])}"
                         f"  eff={e['effective_weight']:.4f}", fontsize=8)
        fig.suptitle(f"Case {case_id} - {title_prefix} effective patches", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)

    _plot(list(reversed(ranked[-k:])), "Top", f"case_{case_id}_top_effective_patches.png")
    _plot(ranked[:k], "Bottom", f"case_{case_id}_bottom_effective_patches.png")


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------
RECORD_FIELDS = [
    "case_id", "stain", "slice_idx", "slice_id", "patch_file", "patch_path", "patch_row",
    "n_patches_in_slice", "patch_attn_weight", "slice_attn_weight", "stain_attn_weight",
    "effective_weight", "true_label", "pred_label",
]


def save_all_patch_records(patch_records, case_label_info, output_dir: str):
    """Every test patch with its filename, full path and attention weights."""
    path = os.path.join(output_dir, "patch_attention_by_filename.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RECORD_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in patch_records:
            w.writerow({**r, **case_label_info[r["case_id"]]})
    print(f"Patch attention by filename saved to: {path}")


def save_attention_summary(attention_summary: List[Dict], output_dir: str):
    summary_path = os.path.join(output_dir, "attention_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\nATTENTION ANALYSIS SUMMARY\n" + "=" * 60 + "\n\n")
        for info in attention_summary:
            f.write(f"Case {info['case_id']}:\n" + "-" * 40 + "\n")
            if info.get("most_attended_stain"):
                f.write(f"Most attended stain: {info['most_attended_stain']}\n")
            f.write("\nStain-level attention:\n")
            for stain, weight in info["stain_attention"].items():
                f.write(f"  {stain}: {weight:.4f}\n")
            f.write("\n")
    print(f"Attention summary saved to: {summary_path}")


def plot_effective_patch_attention_distribution_per_case(patch_records, case_label_info,
                                                         output_dir: str, bins: int = 50):
    if not patch_records:
        print("No effective patch attention data available for per-case plotting.")
        return
    case_to_weights: Dict[Any, List[float]] = defaultdict(list)
    for rec in patch_records:
        case_to_weights[rec["case_id"]].append(rec["effective_weight"])

    for cid, weights_list in case_to_weights.items():
        weights = np.array(weights_list, dtype=np.float32)
        labels = case_label_info[cid]
        counts, edges = np.histogram(weights, bins=bins)
        norm = counts / max(counts.sum(), 1)
        centers = (edges[:-1] + edges[1:]) / 2

        plt.figure(figsize=(8, 5))
        plt.bar(centers, norm, width=np.diff(edges), alpha=0.75, align="center")
        for c, h, n in zip(centers, norm, counts):
            if n:
                plt.text(c, h + 0.01, str(n), ha="center", va="bottom", fontsize=8)
        plt.title(f"Case {cid} - Effective Patch Attention\n"
                  f"True: {labels['true_label']}, Pred: {labels['pred_label']}", fontsize=13)
        plt.xlabel("Effective Patch Attention Weight", fontsize=11)
        plt.ylabel("Normalized Patch Density", fontsize=11)
        plt.grid(axis="y", alpha=0.3)
        plt.text(0.5, -0.12, f"Total patches: {len(weights)}", ha="center", va="top",
                 transform=plt.gca().transAxes, fontsize=10)
        plt.savefig(os.path.join(output_dir, f"effective_patch_attn_distro_case_{cid}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
    print("Per-case effective patch attention distributions saved.")


def analyze_top_effective_patches_per_case(patch_records, case_label_info,
                                           output_dir: str, top_percent: float = 5.0):
    case_to_records: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for rec in patch_records:
        case_to_records[rec["case_id"]].append(rec)

    all_top, per_case = [], {}
    for cid, recs in case_to_records.items():
        weights = np.array([r["effective_weight"] for r in recs], dtype=np.float32)
        total_slices = len({(r["stain"], r["slice_idx"]) for r in recs})
        k = max(1, int(len(recs) * top_percent / 100.0))
        top_recs = [recs[i] for i in np.argsort(weights)[-k:][::-1]]
        lab = case_label_info[cid]
        all_top += [{**r, **lab} for r in top_recs]

        stain_counts, slice_counts = defaultdict(int), defaultdict(int)
        for r in top_recs:
            stain_counts[r["stain"]] += 1
            slice_counts[(r["stain"], r["slice_id"])] += 1
        per_case[cid] = {**lab, "num_total_patches": len(recs), "num_top_patches": len(top_recs),
                         "num_total_slices": total_slices,
                         "top_slice_count_ratio": len(slice_counts) / total_slices if total_slices else 0.0,
                         "stain_counts": dict(stain_counts), "slice_counts": dict(slice_counts)}

    csv_path = os.path.join(output_dir, f"top_effective_patches_per_case_{top_percent:.1f}pct.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RECORD_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_top)
    print(f"Per-case top {top_percent:.1f}% effective patches CSV saved to: {csv_path}")

    summary_path = os.path.join(output_dir, f"top_effective_patches_per_case_summary_{top_percent:.1f}pct.txt")
    with open(summary_path, "w") as f:
        f.write(f"TOP {top_percent:.1f}% EFFECTIVE PATCHES PER CASE\n" + "=" * 60 + "\n\n")
        for cid, info in per_case.items():
            f.write(f"Case {cid}:\n  True label: {info['true_label']}\n  Pred label: {info['pred_label']}\n")
            f.write(f"  # total patches: {info['num_total_patches']}\n")
            f.write(f"  # top patches (per-case): {info['num_top_patches']}\n")
            f.write(f"  # total slices: {info['num_total_slices']}\n")
            f.write(f"  slice ratio: {info['top_slice_count_ratio']:.4f}\n  Top-patch counts by stain:\n")
            for stain, cnt in sorted(info["stain_counts"].items()):
                f.write(f"    {stain}: {cnt}\n")
            f.write("  Top-patch counts by (stain, slice):\n")
            for (stain, sid), cnt in sorted(info["slice_counts"].items()):
                f.write(f"    ({stain}, {sid}): {cnt}\n")
            f.write("\n")
    print(f"Per-case top {top_percent:.1f}% summary saved to: {summary_path}")


def plot_slice_attention_distribution_per_caseandstain(slice_records, case_label_info,
                                                       output_dir: str, bins: int = 30):
    if not slice_records:
        print("No slice-level attention data available for per-case/stain plotting.")
        return
    slice_dir = os.path.join(output_dir, "slice_attention")
    os.makedirs(slice_dir, exist_ok=True)

    by_case: Dict[Any, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in slice_records:
        by_case[rec["case_id"]][rec["stain"]].append(rec["slice_attn_weight"])

    for cid, stain_dict in by_case.items():
        stains = sorted(stain_dict)
        fig, axes = plt.subplots(1, len(stains), figsize=(4.5 * len(stains), 4), squeeze=False)
        for ax, stain in zip(axes[0], stains):
            w = np.sort(np.array(stain_dict[stain], dtype=np.float32))[::-1]
            ax.bar(np.arange(1, len(w) + 1), w, alpha=0.8)
            ax.axhline(1.0 / len(w), linestyle="--", linewidth=1, alpha=0.8)
            ax.text(0.98, 0.95, f"Top slice: {w[0] / w.sum():.2f} of total", ha="right", va="top",
                    transform=ax.transAxes, fontsize=8)
            ax.set_title(f"{stain} (n slices: {len(w)})", fontsize=11)
            ax.set_xlabel("Slice rank (1 = highest attention)", fontsize=9)
            ax.set_ylabel("Slice attention weight", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
        lab = case_label_info[cid]
        fig.suptitle(f"Case {cid} - Slice-Level Attention by Stain\n"
                     f"True: {lab['true_label']}, Pred: {lab['pred_label']}", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0.05, 1, 0.9])
        fig.text(0.5, 0.01, f"Total slices (all stains): {sum(len(v) for v in stain_dict.values())}",
                 ha="center", va="bottom", fontsize=9)
        fig.savefig(os.path.join(slice_dir, f"slice_attn_rankplot_case_{cid}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Per-case & per-stain slice attention rank plots saved to: {slice_dir}")