"""
Attention analysis and visualization utilities (robust to pooled-feature datasets).

- Uses PATCH_PT_DIR to find the exact .pt patch filename for (case, stain, slice_idx, patch_idx),
  then maps it to the PNG with the same basename in PATCH_PNG_DIR (extension swap .pt -> .png).

Expected model return when return_attn_weights=True:
attention_weights = {
  "case_weights": Tensor[num_stains],
  "stain_order": List[str],
  "stain_weights": {
      stain: {
          "slice_weights": Tensor[num_slices],
          "patch_weights": List[Tensor[num_patches_in_slice]],
      }, ...
  }
}
"""

import os
import re
import csv
import glob
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch

from config import IMAGE_CONFIG

# ---------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------
PATCH_PNG_DIR = "/projects/e32998/patches_benign_split"

# [CHANGED/NEW] Option B: pooled embeddings directory (.pt) whose basenames match PNGs
PATCH_PT_DIR = "/projects/e32998/patches_varsize_pooled4096"

# ---------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------
# Old PNG cache (kept as fallback)
_PNG_INDEX_CACHE: Dict[Tuple[Any, str, str], Tuple[List[str], Dict[str, List[str]]]] = {}

# [NEW] PT cache for Option B
_PT_INDEX_CACHE: Dict[Tuple[Any, str, str], Tuple[List[str], Dict[str, List[str]]]] = {}

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

            logits, attention_weights = model(stain_slices, return_attn_weights=True)

            label_tensor = case_data.get("label")
            if isinstance(label_tensor, torch.Tensor):
                true_label = int(label_tensor.detach().view(-1)[0].item())
            else:
                true_label = int(label_tensor)

            logits_for_pred = logits.detach()
            if logits_for_pred.dim() == 1:
                logits_for_pred = logits_for_pred.unsqueeze(0)
            elif logits_for_pred.dim() == 0:
                logits_for_pred = logits_for_pred.view(1, 1)
            pred_label = int(torch.argmax(logits_for_pred, dim=1).view(-1)[0].item())

            case_label_info[case_id] = {"true_label": true_label, "pred_label": pred_label}

            case_summary = analyze_case_attention(
                case_id=case_id,
                stain_slices=stain_slices,
                attention_weights=attention_weights,
                output_dir=attention_dir,
                top_n=top_n,
            )
            attention_summary.append(case_summary)

            patch_records = compute_effective_patch_attention(case_id, attention_weights)
            all_patch_records.extend(patch_records)

            visualize_case_effective_patches(
                case_id=case_id,
                stain_slices=stain_slices,
                patch_records=patch_records,
                output_dir=attention_dir,
                top_n=top_n,
            )

            if "stain_weights" in attention_weights:
                for stain, weights_dict in attention_weights["stain_weights"].items():
                    slice_weights = weights_dict.get("slice_weights", None)
                    if slice_weights is None:
                        continue
                    slice_weights_np = slice_weights.detach().cpu().numpy()
                    for slice_idx, sw in enumerate(slice_weights_np):
                        slice_attention_records.append(
                            {
                                "case_id": case_id,
                                "stain": stain,
                                "slice_idx": int(slice_idx),
                                "slice_attn_weight": float(sw),
                            }
                        )

    save_attention_summary(attention_summary, attention_dir)

    plot_effective_patch_attention_distribution_per_case(
        all_patch_records, case_label_info, plots_dir, bins=50
    )

    plot_slice_attention_distribution_per_caseandstain(
        slice_attention_records, case_label_info, attention_dir, bins=30
    )

    analyze_top_effective_patches_per_case(
        all_patch_records, case_label_info, attention_dir, top_percent=5.0
    )

    print(f"Attention analysis saved to: {attention_dir}")

# ---------------------------------------------------------------------
# Case-level analysis
# ---------------------------------------------------------------------
def analyze_case_attention(
    case_id: Any,
    stain_slices: Dict,
    attention_weights: Dict,
    output_dir: str,
    top_n: int = 5,
) -> Dict:
    patch_attention_dir = os.path.join(output_dir, "patch_attention")
    os.makedirs(patch_attention_dir, exist_ok=True)

    case_summary = {
        "case_id": case_id,
        "stain_attention": {},
        "most_attended_stain": None,
        "stain_order": attention_weights.get("stain_order", []),
    }

    if "case_weights" in attention_weights and "stain_order" in attention_weights:
        case_weights = attention_weights["case_weights"].detach().cpu().numpy()
        stain_order = attention_weights["stain_order"]
        if len(case_weights) == len(stain_order) and len(case_weights) > 0:
            max_idx = int(np.argmax(case_weights))
            case_summary["most_attended_stain"] = stain_order[max_idx]
            case_summary["stain_attention"] = {
                stain: float(weight) for stain, weight in zip(stain_order, case_weights)
            }

    if "stain_weights" in attention_weights:
        for stain, weights_dict in attention_weights["stain_weights"].items():
            slice_weights = weights_dict.get("slice_weights", None)
            patch_weights_list = weights_dict.get("patch_weights", [])

            if slice_weights is None or len(patch_weights_list) == 0:
                continue

            slice_weights_np = slice_weights.detach().cpu().numpy()
            if len(slice_weights_np) == 0:
                continue

            most_attended_slice_idx = int(np.argmax(slice_weights_np))
            least_attended_slice_idx = int(np.argmin(slice_weights_np))

            if len(patch_weights_list) > most_attended_slice_idx:
                patch_weights = patch_weights_list[most_attended_slice_idx].detach().cpu().numpy()
                slice_tensor = _safe_get_slice_tensor(stain_slices, stain, most_attended_slice_idx)

                visualize_patch_attention(
                    case_id=case_id,
                    stain=stain,
                    slice_idx=most_attended_slice_idx,
                    slice_tensor=slice_tensor,
                    patch_weights=patch_weights,
                    output_dir=patch_attention_dir,
                    top_n=top_n,
                    prefix="top",
                )

            if len(patch_weights_list) > least_attended_slice_idx:
                patch_weights = patch_weights_list[least_attended_slice_idx].detach().cpu().numpy()
                slice_tensor = _safe_get_slice_tensor(stain_slices, stain, least_attended_slice_idx)

                visualize_patch_attention(
                    case_id=case_id,
                    stain=stain,
                    slice_idx=least_attended_slice_idx,
                    slice_tensor=slice_tensor,
                    patch_weights=patch_weights,
                    output_dir=patch_attention_dir,
                    top_n=top_n,
                    prefix="bottom",
                )

    return case_summary

def _safe_get_slice_tensor(stain_slices: Dict, stain: str, slice_idx: int) -> Optional[torch.Tensor]:
    try:
        lst = stain_slices.get(stain, [])
        if slice_idx < 0 or slice_idx >= len(lst):
            return None
        return lst[slice_idx]
    except Exception:
        return None

# ---------------------------------------------------------------------
# Shared filename parsing
# ---------------------------------------------------------------------
def _parse_slice_id_from_filename(fname: str) -> Optional[str]:
    """
    Extract slice id token from filenames like:
      case_052_match_1_h&e_patch164.png/.pt
      case_97_unmatched_6_h&e_patch90.png/.pt
      case_83_unmatched2_h&e_patch18.png/.pt

    Returns 'match_1', 'unmatched_6', 'unmatched_2'
    """
    base = os.path.basename(fname)

    m = re.match(r"case_\d+_([a-zA-Z]+_\d+)_", base)
    if m:
        return m.group(1).lower()

    m = re.match(r"case_\d+_([a-zA-Z]+)(\d+)_", base)
    if m:
        return f"{m.group(1).lower()}_{m.group(2)}"

    return None

def _file_contains_stain(fname: str, stain: str) -> bool:
    s = stain.lower()
    b = os.path.basename(fname).lower()
    if s == "h&e":
        return ("h&e" in b) or ("_he_" in b) or ("_h&e_" in b)
    return s in b

def _parse_patch_index_from_png_filename(fname: str) -> Optional[int]:
    base = os.path.basename(fname)
    m = re.search(r"_patch(\d+)\.png$", base, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

# [NEW] .pt patch index parse
def _parse_patch_index_from_pt_filename(fname: str) -> Optional[int]:
    base = os.path.basename(fname)
    m = re.search(r"_patch(\d+)\.pt$", base, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

# ---------------------------------------------------------------------
# [NEW] Option B: index PT files, then map basename to PNG
# ---------------------------------------------------------------------
def _build_pt_index_for_case_stain(
    case_id: Any, stain: str, root_dir: str
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Build (slice_ids_sorted, slice_id_to_pt_files_sorted_by_patchidx) for a case+stain from .pt files.
    Cached in _PT_INDEX_CACHE.

    FIX: handle zero-padded case ids in filenames (e.g., case_052_...) even if case_id is int 52.
    """
    key = (case_id, stain, root_dir)
    if key in _PT_INDEX_CACHE:
        return _PT_INDEX_CACHE[key]

    # --- FIX: try both non-padded and 3-digit padded patterns ---
    patterns: List[str] = [os.path.join(root_dir, f"case_{case_id}_*.pt")]

    # If case_id looks like an int, also try zero-padded (common in your dataset: case_052_...)
    try:
        cid_int = int(case_id)
        patterns.append(os.path.join(root_dir, f"case_{cid_int:03d}_*.pt"))
    except Exception:
        pass

    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    # de-dup (if both patterns match same files somehow)
    files = sorted(set(files))

    # stain filter
    files = [f for f in files if _file_contains_stain(f, stain)]

    slice_id_to_files: Dict[str, List[str]] = defaultdict(list)

    for f in files:
        sid = _parse_slice_id_from_filename(f)
        if sid is None:
            continue
        slice_id_to_files[sid].append(f)

    for sid, flist in slice_id_to_files.items():
        flist_sorted = sorted(
            flist,
            key=lambda x: (
                _parse_patch_index_from_pt_filename(x) is None,
                _parse_patch_index_from_pt_filename(x) or 10**18,
                x,
            ),
        )
        slice_id_to_files[sid] = flist_sorted

    slice_ids_sorted = sorted(slice_id_to_files.keys())

    _PT_INDEX_CACHE[key] = (slice_ids_sorted, dict(slice_id_to_files))
    return _PT_INDEX_CACHE[key]

def _map_pt_to_png_path(pt_path: str) -> str:
    """
    Map /.../patches_pooled4096/<basename>.pt -> /.../patches_benign_split/<basename>.png
    (directory swap + extension swap)
    """
    base = os.path.basename(pt_path)
    if base.lower().endswith(".pt"):
        base = base[:-3] + ".png"
    # if the above didn't work for any reason, do a safer replace:
    if not base.lower().endswith(".png"):
        base = os.path.splitext(os.path.basename(pt_path))[0] + ".png"
    return os.path.join(PATCH_PNG_DIR, base)

# ---------------------------------------------------------------------
# Old PNG index (kept as fallback)
# ---------------------------------------------------------------------
def _build_png_index_for_case_stain(
    case_id: Any, stain: str, root_dir: str
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Build (slice_ids_sorted, slice_id_to_png_files_sorted_by_patchidx) for a case+stain from .png files.
    Cached in _PNG_INDEX_CACHE.

    FIX: handle zero-padded case ids in filenames (e.g., case_052_...) even if case_id is int 52.
    """
    key = (case_id, stain, root_dir)
    if key in _PNG_INDEX_CACHE:
        return _PNG_INDEX_CACHE[key]

    # --- FIX: try both non-padded and 3-digit padded patterns ---
    patterns: List[str] = [os.path.join(root_dir, f"case_{case_id}_*.png")]

    try:
        cid_int = int(case_id)
        patterns.append(os.path.join(root_dir, f"case_{cid_int:03d}_*.png"))
    except Exception:
        pass

    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    files = sorted(set(files))

    # stain filter
    files = [f for f in files if _file_contains_stain(f, stain)]

    slice_id_to_files: Dict[str, List[str]] = defaultdict(list)

    for f in files:
        sid = _parse_slice_id_from_filename(f)
        if sid is None:
            continue
        slice_id_to_files[sid].append(f)

    for sid, flist in slice_id_to_files.items():
        flist_sorted = sorted(
            flist,
            key=lambda x: (
                _parse_patch_index_from_png_filename(x) is None,
                _parse_patch_index_from_png_filename(x) or 10**18,
                x,
            ),
        )
        slice_id_to_files[sid] = flist_sorted

    slice_ids_sorted = sorted(slice_id_to_files.keys())

    _PNG_INDEX_CACHE[key] = (slice_ids_sorted, dict(slice_id_to_files))
    return _PNG_INDEX_CACHE[key]

# ---------------------------------------------------------------------
# [CHANGED] Lookup PNG via PT basename first (Option B), then fallback
# ---------------------------------------------------------------------
def _lookup_png_for_patch(
    case_id: Any, stain: str, slice_idx: int, patch_idx: int, root_dir: str = PATCH_PNG_DIR
) -> Optional[str]:
    """
    Option B (preferred):
      1) Find the .pt file for (case, stain, slice_idx, patch_idx) under PATCH_PT_DIR
      2) Map it to PNG path by basename (.pt -> .png) under PATCH_PNG_DIR

    Fallback:
      use old PNG indexing if PT lookup fails.
    """
    # --- Option B: PT-based mapping ---
    try:
        slice_ids_sorted, slice_map = _build_pt_index_for_case_stain(case_id, stain, PATCH_PT_DIR)
        if 0 <= slice_idx < len(slice_ids_sorted):
            sid = slice_ids_sorted[slice_idx]
            files = slice_map.get(sid, [])
            if 0 <= patch_idx < len(files):
                pt_path = files[patch_idx]
                png_path = _map_pt_to_png_path(pt_path)
                if os.path.exists(png_path):
                    return png_path
    except Exception:
        pass

    # --- Fallback: previous best-effort PNG indexing ---
    try:
        slice_ids_sorted, slice_map = _build_png_index_for_case_stain(case_id, stain, root_dir)
        if slice_idx < 0 or slice_idx >= len(slice_ids_sorted):
            return None
        sid = slice_ids_sorted[slice_idx]
        files = slice_map.get(sid, [])
        if patch_idx < 0 or patch_idx >= len(files):
            return None
        return files[patch_idx]
    except Exception:
        return None

# ---------------------------------------------------------------------
# Visualization (robust)
# ---------------------------------------------------------------------
def visualize_patch_attention(
    case_id: Any,
    stain: str,
    slice_idx: int,
    slice_tensor: Optional[torch.Tensor],
    patch_weights: np.ndarray,
    output_dir: str,
    top_n: int = 5,
    prefix: str = "top",
):
    num_patches = int(len(patch_weights))
    if num_patches <= 0:
        return

    if prefix == "top":
        indices = np.argsort(patch_weights)[-top_n:][::-1]
        title_prefix = "Most"
    else:
        indices = np.argsort(patch_weights)[:top_n]
        title_prefix = "Least"

    indices = indices[: min(top_n, num_patches)]
    if len(indices) == 0:
        return

    n_cols = min(5, len(indices))
    n_rows = (len(indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, pidx in enumerate(indices):
        if i >= len(axes):
            break
        ax = axes[i]
        ax.axis("off")

        img = None
        png_path = _lookup_png_for_patch(case_id, stain, slice_idx, int(pidx), PATCH_PNG_DIR)
        if png_path and os.path.exists(png_path):
            try:
                img = Image.open(png_path).convert("RGB")
                ax.imshow(img)
            except Exception:
                img = None

        if img is None and slice_tensor is not None and isinstance(slice_tensor, torch.Tensor):
            try:
                if slice_tensor.dim() == 4 and slice_tensor.shape[1] in (1, 3):
                    patch = slice_tensor[int(pidx)].detach().cpu().numpy()
                    patch_img = np.transpose(patch, (1, 2, 0))
                    if patch_img.shape[-1] == 1:
                        patch_img = patch_img[..., 0]
                    if patch_img.ndim == 3 and patch_img.shape[-1] == 3:
                        mean = np.array(IMAGE_CONFIG["normalize_mean"])
                        std = np.array(IMAGE_CONFIG["normalize_std"])
                        patch_img = patch_img * std + mean
                        patch_img = np.clip(patch_img, 0, 1)
                    ax.imshow(patch_img)
                elif slice_tensor.dim() == 2:
                    vec = slice_tensor[int(pidx)].detach().cpu().numpy().astype(np.float32)
                    _plot_feature_heatmap(ax, vec)
                else:
                    ax.text(0.5, 0.5, "Unsupported tensor shape", ha="center", va="center", fontsize=9)
            except Exception:
                ax.text(0.5, 0.5, "Render failed", ha="center", va="center", fontsize=9)

        ax.set_title(f"Patch {int(pidx)}\nW: {patch_weights[int(pidx)]:.4f}", fontsize=10)

    for j in range(len(indices), len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Case {case_id} - {stain} - Slice {slice_idx}\n{title_prefix} Attended Patches",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    filename = f"case_{case_id}_{stain}_slice{slice_idx}_{prefix}_patches.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _plot_feature_heatmap(ax, vec: np.ndarray):
    v = vec
    n = int(v.size)
    side = int(np.sqrt(n))
    if side * side == n:
        mat = v.reshape(side, side)
    else:
        mat = v.reshape(1, n)

    m = np.nanmean(mat)
    s = np.nanstd(mat) + 1e-8
    mat_z = (mat - m) / s
    ax.imshow(mat_z, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])

# ---------------------------------------------------------------------
# Effective patch attention
# ---------------------------------------------------------------------
def compute_effective_patch_attention(case_id: Any, attention_weights: Dict) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    if "case_weights" not in attention_weights or "stain_weights" not in attention_weights:
        return records

    case_weights = attention_weights["case_weights"].detach().cpu()
    stain_order = attention_weights.get("stain_order", [])
    stain_weights_dict = attention_weights["stain_weights"]

    for stain_idx, stain in enumerate(stain_order):
        if stain not in stain_weights_dict:
            continue

        weights_dict = stain_weights_dict[stain]
        stain_attn_value = float(case_weights[stain_idx].item())

        slice_weights = weights_dict.get("slice_weights", None)
        patch_weights_list = weights_dict.get("patch_weights", [])

        if slice_weights is None or len(patch_weights_list) == 0:
            continue

        slice_weights = slice_weights.detach().cpu()

        for slice_idx, patch_w in enumerate(patch_weights_list):
            if patch_w is None:
                continue
            patch_w = patch_w.detach().cpu()
            if slice_idx >= slice_weights.numel():
                continue
            slice_attn_value = float(slice_weights[slice_idx].item())

            for patch_idx, pw in enumerate(patch_w):
                patch_attn_value = float(pw.item())
                effective_value = stain_attn_value * slice_attn_value * patch_attn_value
                records.append(
                    {
                        "case_id": case_id,
                        "stain": stain,
                        "slice_idx": int(slice_idx),
                        "patch_idx": int(patch_idx),
                        "patch_attn_weight": patch_attn_value,
                        "slice_attn_weight": slice_attn_value,
                        "stain_attn_weight": stain_attn_value,
                        "effective_weight": effective_value,
                    }
                )
    return records

def visualize_case_effective_patches(
    case_id: Any,
    stain_slices: Dict[str, List[torch.Tensor]],
    patch_records: List[Dict[str, Any]],
    output_dir: str,
    top_n: int = 5,
):
    if not patch_records:
        return

    sorted_records = sorted(patch_records, key=lambda r: r.get("effective_weight", 0.0))
    n_select = min(top_n, len(sorted_records))
    top_entries = list(reversed(sorted_records[-n_select:]))
    bottom_entries = sorted_records[:n_select]

    case_effective_dir = os.path.join(output_dir, "case_effective_patches")
    os.makedirs(case_effective_dir, exist_ok=True)

    def _plot_entries(entries: List[Dict[str, Any]], title_prefix: str, filename: str):
        n = len(entries)
        if n == 0:
            return

        n_cols = min(5, n)
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for ax in axes:
            ax.axis("off")

        for ax, entry in zip(axes, entries):
            stain = entry["stain"]
            slice_idx = int(entry["slice_idx"])
            patch_idx = int(entry["patch_idx"])

            img_shown = False

            png_path = _lookup_png_for_patch(case_id, stain, slice_idx, patch_idx, PATCH_PNG_DIR)
            if png_path and os.path.exists(png_path):
                try:
                    img = Image.open(png_path).convert("RGB")
                    ax.imshow(img)
                    img_shown = True
                except Exception:
                    img_shown = False

            if not img_shown:
                slice_list = stain_slices.get(stain, []) or []
                if slice_idx < len(slice_list):
                    sl = slice_list[slice_idx]
                    try:
                        if isinstance(sl, torch.Tensor) and sl.dim() == 4 and patch_idx < sl.shape[0]:
                            patch = sl[patch_idx].detach().cpu().numpy()
                            patch_img = np.transpose(patch, (1, 2, 0))
                            if patch_img.shape[-1] == 3:
                                mean = np.array(IMAGE_CONFIG["normalize_mean"])
                                std = np.array(IMAGE_CONFIG["normalize_std"])
                                patch_img = patch_img * std + mean
                                patch_img = np.clip(patch_img, 0, 1)
                            ax.imshow(patch_img)
                            img_shown = True
                        elif isinstance(sl, torch.Tensor) and sl.dim() == 2 and patch_idx < sl.shape[0]:
                            vec = sl[patch_idx].detach().cpu().numpy().astype(np.float32)
                            _plot_feature_heatmap(ax, vec)
                            img_shown = True
                    except Exception:
                        img_shown = False

            eff_weight = entry.get("effective_weight", 0.0)
            ax.set_title(f"{stain} s{slice_idx} p{patch_idx}\nEff: {eff_weight:.4f}", fontsize=9)
            ax.axis("off")

        plt.suptitle(f"Case {case_id} - {title_prefix} Effective Patches", fontsize=12, fontweight="bold")
        plt.tight_layout()
        filepath = os.path.join(case_effective_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _plot_entries(top_entries, "Top", f"case_{case_id}_top_effective_patches.png")
    _plot_entries(bottom_entries, "Bottom", f"case_{case_id}_bottom_effective_patches.png")

# ---------------------------------------------------------------------
# Reporting + plots (unchanged)
# ---------------------------------------------------------------------
def save_attention_summary(attention_summary: List[Dict], output_dir: str):
    summary_path = os.path.join(output_dir, "attention_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("ATTENTION ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        for case_info in attention_summary:
            case_id = case_info["case_id"]
            f.write(f"Case {case_id}:\n")
            f.write("-" * 40 + "\n")

            if case_info.get("most_attended_stain"):
                f.write(f"Most attended stain: {case_info['most_attended_stain']}\n")

            if case_info.get("stain_attention"):
                f.write("\nStain-level attention:\n")
                for stain, weight in case_info["stain_attention"].items():
                    f.write(f"  {stain}: {weight:.4f}\n")

            f.write("\n")

    print(f"Attention summary saved to: {summary_path}")

def plot_effective_patch_attention_distribution_per_case(
    patch_records: List[Dict[str, Any]],
    case_label_info: Dict[Any, Dict[str, int]],
    output_dir: str,
    bins: int = 50,
):
    if not patch_records:
        print("No effective patch attention data available for per-case plotting.")
        return

    case_to_weights: Dict[Any, List[float]] = defaultdict(list)
    for rec in patch_records:
        case_to_weights[rec["case_id"]].append(rec["effective_weight"])

    for cid, weights_list in case_to_weights.items():
        if not weights_list:
            continue

        weights = np.array(weights_list, dtype=np.float32)
        total_patch_count = len(weights)

        labels = case_label_info.get(cid, {})
        true_label = labels.get("true_label", None)
        pred_label = labels.get("pred_label", None)

        counts, bin_edges = np.histogram(weights, bins=bins)
        total_patches = counts.sum() if counts.sum() > 0 else 1
        normalized_counts = counts / total_patches
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)

        plt.figure(figsize=(8, 5))
        plt.bar(bin_centers, normalized_counts, width=bin_widths, alpha=0.75, align="center")

        for center, height, count in zip(bin_centers, normalized_counts, counts):
            if count == 0:
                continue
            plt.text(center, height + 0.01, str(count), ha="center", va="bottom", fontsize=8)

        title = f"Case {cid} - Effective Patch Attention"
        if true_label is not None or pred_label is not None:
            title += f"\nTrue: {true_label}, Pred: {pred_label}"
        plt.title(title, fontsize=13)
        plt.xlabel("Effective Patch Attention Weight", fontsize=11)
        plt.ylabel("Normalized Patch Density", fontsize=11)
        plt.grid(axis="y", alpha=0.3)
        plt.text(
            0.5,
            -0.12,
            f"Total patches: {total_patch_count}",
            ha="center",
            va="top",
            transform=plt.gca().transAxes,
            fontsize=10,
        )

        fname = f"effective_patch_attn_distro_case_{cid}.png"
        filepath = os.path.join(output_dir, fname)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

    print("Per-case effective patch attention distributions saved.")

def analyze_top_effective_patches_per_case(
    patch_records: List[Dict[str, Any]],
    case_label_info: Dict[Any, Dict[str, int]],
    output_dir: str,
    top_percent: float = 5.0,
):
    case_to_records: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for rec in patch_records:
        case_to_records[rec["case_id"]].append(rec)

    all_top_records: List[Dict[str, Any]] = []
    per_case_summary = {}

    for cid, recs in case_to_records.items():
        if len(recs) == 0:
            continue

        weights = np.array([r["effective_weight"] for r in recs], dtype=np.float32)
        num_case_patches = len(weights)
        total_slices = len({(r["stain"], r["slice_idx"]) for r in recs})

        k = max(1, int(num_case_patches * top_percent / 100.0))
        top_idx = np.argsort(weights)[-k:]
        top_recs = [recs[i] for i in top_idx]

        label_info = case_label_info.get(cid, {})
        true_label = label_info.get("true_label", None)
        pred_label = label_info.get("pred_label", None)

        for r in top_recs:
            all_top_records.append({**r, "true_label": true_label, "pred_label": pred_label})

        stain_counts = defaultdict(int)
        slice_counts = defaultdict(int)

        for r in top_recs:
            stain_counts[r["stain"]] += 1
            slice_counts[(r["stain"], r["slice_idx"])] += 1

        slice_count_ratio = (len(slice_counts) / total_slices) if total_slices > 0 else 0.0

        per_case_summary[cid] = {
            "true_label": true_label,
            "pred_label": pred_label,
            "num_total_patches": num_case_patches,
            "num_top_patches": len(top_recs),
            "num_total_slices": total_slices,
            "top_slice_count_ratio": slice_count_ratio,
            "stain_counts": dict(stain_counts),
            "slice_counts": dict(slice_counts),
        }

    csv_path = os.path.join(output_dir, f"top_effective_patches_per_case_{top_percent:.1f}pct.csv")
    fieldnames = [
        "case_id",
        "stain",
        "slice_idx",
        "patch_idx",
        "patch_attn_weight",
        "slice_attn_weight",
        "stain_attn_weight",
        "effective_weight",
        "true_label",
        "pred_label",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in all_top_records:
            writer.writerow(rec)

    print(f"Per-case top {top_percent:.1f}% effective patches CSV saved to: {csv_path}")

    summary_path = os.path.join(output_dir, f"top_effective_patches_per_case_summary_{top_percent:.1f}pct.txt")
    with open(summary_path, "w") as f:
        f.write(f"TOP {top_percent:.1f}% EFFECTIVE PATCHES PER CASE\n")
        f.write("=" * 60 + "\n\n")

        for cid, info in per_case_summary.items():
            f.write(f"Case {cid}:\n")
            f.write(f"  True label: {info['true_label']}\n")
            f.write(f"  Pred label: {info['pred_label']}\n")
            f.write(f"  # total patches: {info['num_total_patches']}\n")
            f.write(f"  # top patches (per-case): {info['num_top_patches']}\n")
            f.write(f"  # total slices: {info['num_total_slices']}\n")
            f.write(f"  slice ratio: {info['top_slice_count_ratio']:.4f}\n")

            f.write("  Top-patch counts by stain:\n")
            for stain, cnt in sorted(info["stain_counts"].items()):
                f.write(f"    {stain}: {cnt}\n")

            f.write("  Top-patch counts by (stain, slice_idx):\n")
            for (stain, s_idx), cnt in sorted(info["slice_counts"].items(), key=lambda x: (x[0][0], x[0][1])):
                f.write(f"    ({stain}, slice {s_idx}): {cnt}\n")

            f.write("\n")

    print(f"Per-case top {top_percent:.1f}% summary saved to: {summary_path}")

def plot_slice_attention_distribution_per_caseandstain(
    slice_records: List[Dict[str, Any]],
    case_label_info: Dict[Any, Dict[str, int]],
    output_dir: str,
    bins: int = 30,
):
    if not slice_records:
        print("No slice-level attention data available for per-case/stain plotting.")
        return

    slice_dir = os.path.join(output_dir, "slice_attention")
    os.makedirs(slice_dir, exist_ok=True)

    case_to_stain_to_weights: Dict[Any, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in slice_records:
        case_to_stain_to_weights[rec["case_id"]][rec["stain"]].append(rec["slice_attn_weight"])

    for cid, stain_dict in case_to_stain_to_weights.items():
        if not stain_dict:
            continue

        stains = sorted(stain_dict.keys())
        num_stains = len(stains)

        fig, axes = plt.subplots(1, num_stains, figsize=(4.5 * num_stains, 4), squeeze=False)
        axes = axes[0]

        labels = case_label_info.get(cid, {})
        true_label = labels.get("true_label", None)
        pred_label = labels.get("pred_label", None)

        for ax, stain in zip(axes, stains):
            weights_list = stain_dict[stain]
            if not weights_list:
                ax.set_visible(False)
                continue

            weights = np.array(weights_list, dtype=np.float32)
            n_slices = len(weights)

            sorted_w = np.sort(weights)[::-1]
            ranks = np.arange(1, n_slices + 1)
            uniform_level = 1.0 / n_slices

            ax.bar(ranks, sorted_w, alpha=0.8)
            ax.axhline(uniform_level, linestyle="--", linewidth=1, alpha=0.8)

            top1_share = sorted_w[0] / sorted_w.sum() if sorted_w.sum() > 0 else 0.0
            ax.text(
                0.98,
                0.95,
                f"Top slice: {top1_share:.2f} of total",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=8,
            )

            ax.set_title(f"{stain} (n slices: {n_slices})", fontsize=11)
            ax.set_xlabel("Slice rank (1 = highest attention)", fontsize=9)
            ax.set_ylabel("Slice attention weight", fontsize=9)
            ax.grid(axis="y", alpha=0.3)

        title = f"Case {cid} - Slice-Level Attention by Stain"
        if true_label is not None or pred_label is not None:
            title += f"\nTrue: {true_label}, Pred: {pred_label}"

        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0.05, 1, 0.9])

        total_slices_case = sum(len(w) for w in stain_dict.values())
        fig.text(0.5, 0.01, f"Total slices (all stains): {total_slices_case}", ha="center", va="bottom", fontsize=9)

        fname = f"slice_attn_rankplot_case_{cid}.png"
        filepath = os.path.join(slice_dir, fname)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Per-case & per-stain slice attention rank plots saved to: {slice_dir}")