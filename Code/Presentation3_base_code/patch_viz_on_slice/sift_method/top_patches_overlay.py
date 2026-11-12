#!/usr/bin/env python3
"""
top_patches_overlay.py — finds the top-k or 5 attention patches of each slice on
that slice, colored corresponding to attention weight. It produces one image per slice.

For each requested case it reads the attention_analysis CSV, finds every
(stain, slice) the given case has, keeps the K highest-attention patches of each
slice, locates each patch on the slice with find_patch (SIFT + RANSAC), and
draws all K outlines on one downscaled copy of the slice with a legend.

Usage (command line):
    python top_patches_overlay.py --run-dir /projects/e32998/MIL_training/final_runs/run_20261022_115349 \
        --cases 5 7 [--k 5] [--slice-root /path/to/full_slice_tifs]

Usage (notebook):
    from top_patches_overlay import run
    images = run("/projects/e32998/MIL_training/final_runs/run_20261103_164248", [5, 7])

Outputs, per slice, in OUT/case_<n>/:
    case<n>_<stain>_slice<i>_top<K>.png      the overlay (the figure you want)
    case<n>_<stain>_slice<i>_checks.png      each patch next to the region it
                                             was matched to, for a visual check
and OUT/match_summary.csv with one row per patch (location, rotation,
inliers, confidence).

Requires find_patch.py in the same folder, plus opencv-python, numpy, pandas, pillow.
"""

import argparse
import glob
import os
import re
import sys
 
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
 
import find_patch as fp
from find_patch import build_sift, gray, load_rgb
 
# OpenCV's BFMatcher.knnMatch refuses more than 2**18 - 1 (262,143) train
# descriptors, and a full-resolution slice has millions. ChunkedBFMatcher splits
# the slice descriptors into chunks, matches each, and keeps the k best per query
# overall, which is exactly what one big BFMatcher call would return.
MATCH_CHUNK = 200_000
 
 
class ChunkedBFMatcher:
    def knnMatch(self, query, train, k=2):
        if len(train) <= MATCH_CHUNK:
            return cv2.BFMatcher().knnMatch(query, train, k=k)
        best = [[] for _ in range(len(query))]
        bf = cv2.BFMatcher()
        for off in range(0, len(train), MATCH_CHUNK):
            for qi, ms in enumerate(bf.knnMatch(query, train[off:off + MATCH_CHUNK], k=k)):
                best[qi].extend(cv2.DMatch(m.queryIdx, m.trainIdx + off, m.distance) for m in ms)
        return [sorted(ms, key=lambda m: m.distance)[:k] for ms in best]
 
 
class _CV2WithChunkedMatcher:
    """Stands in for cv2 inside find_patch.py so its BFMatcher() is the chunked one."""
    def __getattr__(self, name):
        return ChunkedBFMatcher if name == "BFMatcher" else getattr(cv2, name)
 
 
fp.cv2 = _CV2WithChunkedMatcher()
find_patch = fp.find_patch
 
# ---------------------------------------------------------------------------
# CONFIG — edit these to match where the slices and patches live on Quest.
# ---------------------------------------------------------------------------
# The attention CSV is looked up inside the run directory you pass in:
#   <run_dir>/attention_analysis/<CSV_NAME>
CSV_NAME = "top_effective_patches_per_case_5.0pct.csv"
# Folder holding one subfolder per case (Case_083, ...). Each case folder holds
# either match_1/, match_2/ ... subfolders with one .tif per stain, or the
# stain .tifs directly (then ROI_<n> in the name is taken as match <n>).
# EDIT THIS to the real location of full_slice_tifs (or pass slice_root= / --slice-root).
SLICE_ROOT = "/projects/e32998/STAT390_Fall2026/Team1_FullStack/Presentation2_base_code/full_slice_tifs"
# Each .tif is matched to the slice named in the CSV's patch_filename column:
# match_<n>/ subfolder (or ROI_<n>) <-> "match_<n>" in case_83_match_1_melan_patch*.png.
# The CSV's slice_idx/patch_idx are positions, NOT the numbers in file names.
# Top patches from slices without a .tif (other matches, "unmatched") are listed and skipped.
# Folder of patches named like case_1_match_1_h&e_patch0.png
PATCH_DIR = "/projects/e32998/patches_benign_split"
 
# Value in the CSV's `stain` column -> spelling used in file names.
# Stains not listed here are used as-is.
STAIN_TOKENS = {
    # "HE": "h&e",
    # "MelanA": "melan",
    # "SOX10": "sox10",
}
 
# How CSV `patch_idx` maps to a patch file:
#   "filename"        patch_idx is the number in the file name (patch32 -> 32)
#   "sorted_numeric"  patch_idx is the position in the slice's patch list sorted by number
#   "sorted_alpha"    patch_idx is the position in the slice's patch list sorted alphabetically
# Check this against attention_analysis.py before trusting the figures.
PATCH_INDEX_MODE = "filename"
 
# Rank colors (RGB), hottest = highest attention, plus line widths in output pixels.
RANK_COLORS = [(230, 25, 25), (255, 140, 0), (255, 220, 0), (40, 190, 80), (30, 140, 255)]
RANK_WIDTHS = [7, 6, 5, 4, 4]
OUTPUT_MAX_WIDTH = 1600  # overlay images are downscaled to at most this width
# ---------------------------------------------------------------------------
 
 
def case_number(value):
    m = re.search(r"\d+", str(value))
    if m is None:
        raise ValueError(f"Cannot read a case number from {value!r}")
    return int(m.group())
 
 
def stain_token(stain):
    return STAIN_TOKENS.get(stain, stain)
 
 
# Patch file names are parsed rather than built from a template, because the
# data has variants like case_063_... (zero-padded) and case_11_unmatched_1_...
PATCH_RE = re.compile(r"^case_0*(\d+)_(match|unmatched)_0*(\d+)_(.+)_patch0*(\d+)\.(png|tif|tiff|jpg|jpeg)$", re.I)
CASE_DIR_RE = re.compile(r"^case[_ ]?0*(\d+)$", re.I)
SLICE_TIF_RE = re.compile(r"(h&e|melan|sox10)[_ ]*ROI[_ ]*0*(\d+)\.tiff?$", re.I)
_index_cache = {}
 
 
def _file_index(folder, regex, n_keys):
    """{(case, slice, stain[, patch]): [paths]} for every file in folder whose name matches regex."""
    cache_key = (os.path.abspath(folder), regex.pattern)
    if cache_key not in _index_cache:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        idx = {}
        for name in os.listdir(folder):
            m = regex.match(name)
            if m:
                g = m.groups()
                key = (int(g[0]), int(g[2]), g[3].lower()) + ((int(g[4]),) if n_keys == 4 else ())
                idx.setdefault(key, []).append(os.path.join(folder, name))
        _index_cache[cache_key] = idx
    return _index_cache[cache_key]
 
 
def _pick(paths, what):
    """Prefer 'match' over 'unmatched' if both exist; raise if still ambiguous."""
    if len(paths) > 1:
        matched = [p for p in paths if "_unmatched_" not in os.path.basename(p).lower()]
        if len(matched) == 1:
            return matched[0]
        raise ValueError(f"More than one file for {what}: {paths}")
    return paths[0]
 
 
MATCH_DIR_RE = re.compile(r"^match[_ ]?0*(\d+)$", re.I)
SLICE_TIF_ANY_RE = re.compile(r"(h&e|melan|sox10).*\.tiff?$", re.I)
 
 
def slice_index(slice_root):
    """{(case, stain, match): path} for the slice .tifs under <slice_root>/Case_XXX/.
 
    Two layouts are understood:
      Case_083/match_1/<...> melan_ROI_1.tif   match number from the subfolder name
      Case_005/<...> melan_ROI_1.tif           match number from ROI_<n> in the file name
    """
    if not os.path.isdir(slice_root):
        raise FileNotFoundError(f"Slice folder not found: {slice_root}  (edit SLICE_ROOT or pass slice_root=)")
    idx = {}
 
    def add(case, stain, n, path):
        key = (case, stain.lower(), n)
        if key in idx:
            raise ValueError(f"Two slice files for case {case} {stain} match {n}: {idx[key]} and {path}")
        idx[key] = path
 
    for d in sorted(os.listdir(slice_root)):
        m = CASE_DIR_RE.match(d)
        case_dir = os.path.join(slice_root, d)
        if not m or not os.path.isdir(case_dir):
            continue
        case = int(m.group(1))
        for name in sorted(os.listdir(case_dir)):
            path = os.path.join(case_dir, name)
            sub = MATCH_DIR_RE.match(name)
            if sub and os.path.isdir(path):
                for f in sorted(os.listdir(path)):
                    t = SLICE_TIF_ANY_RE.search(f)
                    if t:
                        add(case, t.group(1), int(sub.group(1)), os.path.join(path, f))
            else:
                t = SLICE_TIF_RE.search(name)
                if t:
                    add(case, t.group(1), int(t.group(2)), path)
    return idx
 
 
def tif_key_from_filename(case, stain, filename):
    """(case, stain, n) for '..._match_<n>_<stain>_patch..'; None for unmatched/odd names."""
    m = re.search(r"_match_0*(\d+)_", str(filename))
    return (case, stain_token(stain).lower(), int(m.group(1))) if m else None
 
 
def load_slice(path):
    """Load a slice as RGB uint8; falls back to tifffile for TIFFs PIL can't open."""
    try:
        return load_rgb(path)
    except Exception as e:
        try:
            import tifffile
        except ImportError:
            raise RuntimeError(f"PIL could not open {path} ({e}); try `pip install tifffile`") from e
        img = tifffile.imread(path)
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        img = img[..., :3]
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8)
        return np.ascontiguousarray(img)
 
 
def patch_path(case, stain, slice_idx, patch_idx):
    idx = _file_index(PATCH_DIR, PATCH_RE, 4)
    base = (int(case), int(slice_idx), stain_token(stain).lower())
    if PATCH_INDEX_MODE == "filename":
        paths = idx.get(base + (int(patch_idx),))
        if not paths:
            raise FileNotFoundError(f"No patch file for case {case}, {stain}, slice {slice_idx}, "
                                    f"patch {patch_idx} in {PATCH_DIR}")
        return _pick(paths, f"patch {base + (patch_idx,)}")
 
    files = [_pick(v, str(k)) for k, v in idx.items() if k[:3] == base]
    if not files:
        raise FileNotFoundError(f"No patch files for case {case}, {stain}, slice {slice_idx} in {PATCH_DIR}")
    if PATCH_INDEX_MODE == "sorted_alpha":
        files.sort()
    elif PATCH_INDEX_MODE == "sorted_numeric":
        files.sort(key=lambda f: int(re.findall(r"\d+", os.path.basename(f))[-1]))
    else:
        raise ValueError(f"Unknown PATCH_INDEX_MODE {PATCH_INDEX_MODE!r}")
    if not 0 <= int(patch_idx) < len(files):
        raise IndexError(f"patch_idx {patch_idx} out of range: {len(files)} patches for case {case}, {stain}, slice {slice_idx}")
    return files[int(patch_idx)]
 
 
def select_top_patches(df, cases, k, max_slices_per_stain=None):
    """Top-k rows per (case, stain, slice), ranked by effective_weight."""
    df = df.copy()
    df["case_num"] = df["case_id"].map(case_number)
    missing = sorted(set(cases) - set(df["case_num"]))
    if missing:
        available = sorted(df["case_num"].unique())
        raise ValueError(f"Case(s) {missing} are not in the CSV. Cases it has ({len(available)}): {available}\n"
                         f"Raw case_id examples: {list(df['case_id'].unique()[:5])}")
    df = df[df["case_num"].isin(cases)]
 
    if max_slices_per_stain:
        # keep the slices the model weighted most heavily within each stain
        slices = (df.groupby(["case_num", "stain", "slice_idx"])["slice_attn_weight"].first()
                    .reset_index().sort_values("slice_attn_weight", ascending=False)
                    .groupby(["case_num", "stain"]).head(max_slices_per_stain))
        df = df.merge(slices[["case_num", "stain", "slice_idx"]])
 
    keys = ["case_num", "stain", "slice_idx"]
    df = df.sort_values("effective_weight", ascending=False)
    top = df.groupby(keys, sort=False).head(k).copy()
    top["rank"] = top.groupby(keys, sort=False).cumcount() + 1
    return top.sort_values(keys + ["rank"])
 
 
def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()
 
 
def draw_overlay(slice_rgb, matches, title, k):
    """matches: list of dicts with rank, patch_idx, weight, res (find_patch output or None)."""
    f = max(1.0, slice_rgb.shape[1] / OUTPUT_MAX_WIDTH)
    small = cv2.resize(slice_rgb, (round(slice_rgb.shape[1] / f), round(slice_rgb.shape[0] / f)),
                       interpolation=cv2.INTER_AREA)
 
    # Draw lowest rank first so the top patch sits on top where boxes overlap.
    for m in sorted(matches, key=lambda m: -m["rank"]):
        if m["res"] is None:
            continue
        i = m["rank"] - 1
        pts = (np.array(m["res"]["corners"]) / f).round().astype(np.int32)
        cv2.polylines(small, [pts], True, (0, 0, 0), RANK_WIDTHS[i] + 4, cv2.LINE_AA)   # dark halo
        cv2.polylines(small, [pts], True, RANK_COLORS[i], RANK_WIDTHS[i], cv2.LINE_AA)
 
    img = Image.fromarray(small)
    d = ImageDraw.Draw(img)
    label_font = _font(max(16, small.shape[1] // 45))
    for m in matches:
        if m["res"] is None:
            continue
        pts = np.array(m["res"]["corners"]) / f
        x, y = pts[np.argmin(pts[:, 1])]  # label at the box's top corner
        d.text((x + 6, y - 6), str(m["rank"]), font=label_font, fill=RANK_COLORS[m["rank"] - 1],
               stroke_width=3, stroke_fill=(0, 0, 0), anchor="ls")
 
    # Legend panel on the right.
    fs = max(14, small.shape[1] // 55)
    font, title_font = _font(fs), _font(int(fs * 1.15))
    row_h = int(fs * 1.9)
    lines = title.split("\n")
    widest = max([title_font.getlength(l) for l in lines] + [font.getlength("#5  00000  0.00000  NOT FOUND") + fs * 1.6])
    panel_w = int(widest + fs * 2)
    panel_h = max(img.height, row_h * (len(lines) + k + 3))
    canvas = Image.new("RGB", (img.width + panel_w, panel_h), "white")
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    x0, y = img.width + fs, fs
    for line in lines:
        d.text((x0, y), line, font=title_font, fill="black")
        y += row_h
    y += row_h // 2
    d.text((x0, y), "Rank  patch#  effective wt", font=font, fill=(90, 90, 90))
    y += row_h
    for m in matches:
        c = RANK_COLORS[m["rank"] - 1]
        d.rectangle([x0, y + 2, x0 + fs, y + fs + 2], fill=c, outline="black", width=2)
        status = "" if m["res"] is None or m["res"]["confidence"] != "low" else "  (low conf.)"
        if m["res"] is None:
            status = "  NOT FOUND"
        d.text((x0 + int(fs * 1.6), y), f"#{m['rank']}  {m['patch_idx']:>5}  {m['weight']:.4g}{status}",
               font=font, fill="black")
        y += row_h
    return canvas
 
 
def draw_checks(slice_rgb, matches, size=224):
    """Each patch beside the slice region it was matched to."""
    rows = []
    for m in matches:
        patch = cv2.resize(m["patch"], (size, size))
        if m["res"] is None:
            region = np.full_like(patch, 255)
        else:
            h, w = m["patch"].shape[:2]
            region = cv2.warpAffine(slice_rgb, cv2.invertAffineTransform(m["res"]["H"]), (w, h))
            region = cv2.resize(region, (size, size))
        bar = np.zeros((size, 16, 3), np.uint8)
        bar[:] = RANK_COLORS[m["rank"] - 1]
        rows.append(np.hstack([bar, patch, np.full((size, 8, 3), 255, np.uint8), region]))
        rows.append(np.full((8, rows[-1].shape[1], 3), 255, np.uint8))
    return Image.fromarray(np.vstack(rows[:-1]))
 
 
def attention_csv(run_dir, csv_name=None):
    path = os.path.join(os.path.expanduser(run_dir.strip()), "attention_analysis", csv_name or CSV_NAME)
    if not os.path.exists(path):
        found = glob.glob(os.path.join(os.path.expanduser(run_dir.strip()), "attention_analysis", "*.csv"))
        raise FileNotFoundError(f"Attention CSV not found: {path}\n"
                                f"CSVs in that folder: {[os.path.basename(f) for f in found] or 'none'}")
    return path
 
 
def run(run_dir, cases, k=5, out=None, csv_name=None, slice_root=None, max_slices_per_stain=None):
    """Make one overlay per slice .tif for each case. Returns the overlay file paths.
 
    run_dir     model run folder, e.g. "/projects/e32998/MIL_training/final_runs/run_20261103_164248"
    cases       case numbers, e.g. [5, 7] or ["005", "Case_007"]
    out         output folder (default: top_patch_overlays/<run name>)
    slice_root  folder with Case_XXX subfolders of .tifs (default: SLICE_ROOT)
    """
    cases = [case_number(c) for c in cases]
    if k > len(RANK_COLORS):
        raise ValueError(f"k={k} but only {len(RANK_COLORS)} rank colors are defined")
    csv_path = attention_csv(run_dir, csv_name)
    print(f"Using attention CSV {csv_path}", file=sys.stderr)
    out = out or os.path.join("top_patch_overlays", os.path.basename(os.path.normpath(run_dir.strip())))
    df = pd.read_csv(csv_path)
    top = select_top_patches(df, cases, k, max_slices_per_stain)
 
    # Keep only slices that have a .tif; say which ones are skipped.
    tifs = slice_index(slice_root or SLICE_ROOT)
    if "patch_filename" in top.columns:
        top["tif_key"] = [tif_key_from_filename(c, s, f) for c, s, f in
                          zip(top["case_num"], top["stain"], top["patch_filename"])]
    else:
        top["tif_key"] = [(c, stain_token(s).lower(), int(i)) for c, s, i in
                          zip(top["case_num"], top["stain"], top["slice_idx"])]
    for case in cases:
        have = sorted(k_[1:] for k_ in tifs if k_[0] == case)
        if not have:
            raise FileNotFoundError(f"No Case_{case:03d} folder with stain .tifs under {slice_root or SLICE_ROOT}")
        print(f"Case {case}: slice .tifs found for (stain, match) {have}", file=sys.stderr)
    skipped = top[~top["tif_key"].isin(tifs)].groupby(["case_num", "stain", "slice_idx"]).size()
    for (c, s_, i), n in skipped.items():
        names = top[(top.case_num == c) & (top.stain == s_) & (top.slice_idx == i)]
        label = names["patch_filename"].iloc[0].rsplit("_patch", 1)[0] if "patch_filename" in names else f"slice {i}"
        print(f"SKIPPED case {c} {s_} slice_idx {i} ({label}): {n} top patch(es), no matching .tif")
    top = top[top["tif_key"].isin(tifs)]
    if top.empty:
        raise ValueError("None of the top patches belong to a slice with a .tif - check the ROI numbering")
 
    sift = build_sift()
    written, summary = [], []
    for (case, stain, slice_idx), rows in top.groupby(["case_num", "stain", "slice_idx"], sort=False):
        tag = f"case{case}_{stain_token(stain)}_match{rows['tif_key'].iloc[0][2]}"
        if len(rows) < k:
            print(f"WARNING {tag}: only {len(rows)} patches of this slice are in the CSV (wanted {k})")
        spath = tifs[rows["tif_key"].iloc[0]]
        print(f"{tag}: loading {spath} and extracting features ...", file=sys.stderr)
        slice_rgb = load_slice(spath)
        slice_gray = gray(slice_rgb)
        skp, sdesc = sift.detectAndCompute(slice_gray, None)
 
        matches = []
        for r in rows.itertuples():
            if "patch_filename" in rows.columns:
                ppath = os.path.join(PATCH_DIR, r.patch_filename)
                if not os.path.exists(ppath):
                    raise FileNotFoundError(f"Patch file from CSV not found: {ppath}")
            else:
                ppath = patch_path(case, stain, slice_idx, r.patch_idx)
            patch = load_rgb(ppath)
            res = find_patch(slice_gray, skp, sdesc, patch, sift)
            fnum = re.search(r"patch0*(\d+)\.\w+$", ppath)
            matches.append(dict(rank=r.rank, patch_idx=int(fnum.group(1)) if fnum else r.patch_idx, weight=r.effective_weight,
                                patch=patch, res=res))
            if res is None:
                print(f"  #{r.rank} {os.path.basename(ppath)}: NOT FOUND ({ppath})")
            else:
                cx, cy = res["center"]
                print(f"  #{r.rank} {os.path.basename(ppath)}: center=({cx:.0f},{cy:.0f}) "
                      f"rot={res['rotation_deg']:.1f}deg scale={res['scale']:.3f} "
                      f"inliers={res['inliers']} ({res['confidence']})")
            summary.append(dict(case=case, stain=stain, slice_idx=slice_idx, rank=r.rank,
                                patch_idx=r.patch_idx, effective_weight=r.effective_weight,
                                patch_file=ppath, found=res is not None,
                                center_x=None if res is None else round(res["center"][0], 1),
                                center_y=None if res is None else round(res["center"][1], 1),
                                rotation_deg=None if res is None else round(res["rotation_deg"], 2),
                                scale=None if res is None else round(res["scale"], 4),
                                inliers=None if res is None else res["inliers"],
                                confidence=None if res is None else res["confidence"]))
 
        first = rows.iloc[0]
        title = (f"Case {case} | {stain} | {os.path.basename(spath)}\n"
                 f"true: {first.true_label}   pred: {first.pred_label}")
        case_dir = os.path.join(out, f"case_{case}")
        os.makedirs(case_dir, exist_ok=True)
        overlay_path = os.path.join(case_dir, f"{tag}_top{k}.png")
        draw_overlay(slice_rgb, matches, title, k).save(overlay_path)
        draw_checks(slice_rgb, matches).save(os.path.join(case_dir, f"{tag}_checks.png"))
        written.append(overlay_path)
 
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(summary).to_csv(os.path.join(out, "match_summary.csv"), index=False)
    not_found = [s for s in summary if not s["found"]]
    if not_found:
        print(f"\n{len(not_found)} patch(es) NOT FOUND - see match_summary.csv")
    print(f"Wrote {len(written)} overlay(s) to {out}/", file=sys.stderr)
    return written
 
 
def main():
    ap = argparse.ArgumentParser(description="Outline the top-K attention patches on each slice.")
    ap.add_argument("--run-dir", required=True, help="model run folder containing attention_analysis/")
    ap.add_argument("--cases", nargs="+", required=True, help="case numbers, e.g. 20 34")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None, help="output folder (default: top_patch_overlays/<run name>)")
    ap.add_argument("--slice-root", default=None, help=f"folder of Case_XXX .tif folders (default: {SLICE_ROOT})")
    ap.add_argument("--csv-name", default=None, help=f"CSV file name inside attention_analysis/ (default: {CSV_NAME})")
    ap.add_argument("--max-slices-per-stain", type=int, default=None,
                    help="keep only the N slices per stain with the highest slice_attn_weight")
    a = ap.parse_args()
    run(a.run_dir, a.cases, a.k, a.out, a.csv_name, a.slice_root, a.max_slices_per_stain)
 
 
if __name__ == "__main__":
    main()