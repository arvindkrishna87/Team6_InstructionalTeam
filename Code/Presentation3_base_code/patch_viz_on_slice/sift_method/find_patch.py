#!/usr/bin/env python3
"""
find_patch.py — locate one or more image patches on a (histology) slice.

Uses SIFT feature matching + RANSAC similarity transform, so it handles
patches that are rotated, scaled, or have a different color cast than the slice.

Usage:
    python find_patch.py SLICE PATCH [PATCH ...] [--out DIR] [--no-images]

Example:
    python find_patch.py case_20_match_1_h_e.tif patch1.png patch2.png --out results

Outputs (per patch, in --out):
    <name>_overview.png   slice (downscaled) with the patch outlined
    <name>_closeup.png    full-res crop around the match
    <name>_compare.png    patch next to the region warped out of the slice
    results.json / results.csv   center, corners, scale, rotation, inliers, confidence

Requires: opencv-python (cv2), numpy, pillow
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # allow very large slices


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def build_sift():
    return cv2.SIFT_create(nfeatures=0, contrastThreshold=0.02)


def find_patch(slice_gray, slice_kp, slice_desc, patch_rgb, sift, ratio=0.8, reproj=5.0):
    """Return dict with transform and quality stats, or None if no match."""
    h, w = patch_rgb.shape[:2]
    best = None
    # Upsample small patches a bit so SIFT finds enough keypoints.
    for up in (1, 2):
        p = patch_rgb if up == 1 else cv2.resize(patch_rgb, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        kp1, d1 = sift.detectAndCompute(gray(p), None)
        if d1 is None or len(kp1) < 4:
            continue
        matches = cv2.BFMatcher().knnMatch(d1, slice_desc, k=2)
        good = [m for m, n in (pair for pair in matches if len(pair) == 2) if m.distance < ratio * n.distance]
        if len(good) < 4:
            continue
        src = np.float32([kp1[g.queryIdx].pt for g in good]) / up  # back to patch pixel coords
        dst = np.float32([slice_kp[g.trainIdx].pt for g in good])
        H, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=reproj)
        if H is None:
            continue
        inliers = int(mask.sum())
        if best is None or inliers > best["inliers"]:
            best = {"H": H, "inliers": inliers, "good": len(good), "kps": len(kp1)}
    if best is None:
        return None

    H = best["H"]
    scale = float(np.hypot(H[0, 0], H[0, 1]))
    angle = float(np.degrees(np.arctan2(-H[1, 0], H[0, 0])))  # clockwise positive
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    c = cv2.transform(corners, H).reshape(-1, 2)
    center = c.mean(0)
    inl = best["inliers"]
    confidence = "high" if inl >= 50 else "medium" if inl >= 15 else "low"
    return {
        "H": H, "scale": scale, "rotation_deg": angle,
        "center": center.tolist(), "corners": c.tolist(),
        "bbox": [float(c[:, 0].min()), float(c[:, 1].min()), float(c[:, 0].max()), float(c[:, 1].max())],
        "inliers": inl, "good_matches": best["good"], "patch_keypoints": best["kps"],
        "confidence": confidence,
    }


def save_images(slice_rgb, patch_rgb, res, name, out_dir):
    c = np.array(res["corners"], dtype=np.float32)
    H = res["H"]
    # overview
    ov = slice_rgb.copy()
    cv2.polylines(ov, [c.astype(np.int32)], True, (0, 255, 0), max(4, slice_rgb.shape[1] // 400))
    f = max(1, slice_rgb.shape[1] // 800)
    Image.fromarray(ov).resize((ov.shape[1] // f, ov.shape[0] // f)).save(os.path.join(out_dir, f"{name}_overview.png"))
    # closeup
    pad = int(max(c.max(0) - c.min(0)) * 0.4) + 20
    x0, y0 = np.maximum(c.min(0) - pad, 0).astype(int)
    x1, y1 = np.minimum(c.max(0) + pad, [slice_rgb.shape[1], slice_rgb.shape[0]]).astype(int)
    cr = slice_rgb[y0:y1, x0:x1].copy()
    cv2.polylines(cr, [(c - [x0, y0]).astype(np.int32)], True, (0, 255, 0), 3)
    Image.fromarray(cr).save(os.path.join(out_dir, f"{name}_closeup.png"))
    # compare: patch vs region warped out of the slice
    h, w = patch_rgb.shape[:2]
    warped = cv2.warpAffine(slice_rgb, cv2.invertAffineTransform(H), (w, h))
    Image.fromarray(np.hstack([patch_rgb, warped])).save(os.path.join(out_dir, f"{name}_compare.png"))


def main():
    ap = argparse.ArgumentParser(description="Locate patches on a slice image.")
    ap.add_argument("slice")
    ap.add_argument("patches", nargs="+")
    ap.add_argument("--out", default="patch_results")
    ap.add_argument("--no-images", action="store_true", help="only write results.json/csv")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading slice {args.slice} ...", file=sys.stderr)
    slice_rgb = load_rgb(args.slice)
    sift = build_sift()
    print("Extracting slice features (one-time) ...", file=sys.stderr)
    skp, sdesc = sift.detectAndCompute(gray(slice_rgb), None)
    print(f"  {len(skp)} keypoints", file=sys.stderr)

    results = []
    for p in args.patches:
        name = os.path.splitext(os.path.basename(p))[0]
        patch = load_rgb(p)
        res = find_patch(gray(slice_rgb), skp, sdesc, patch, sift)
        if res is None:
            print(f"{name}: NOT FOUND")
            results.append({"patch": p, "found": False})
            continue
        cx, cy = res["center"]
        print(f"{name}: center=({cx:.0f},{cy:.0f}) scale={res['scale']:.3f} "
              f"rot={res['rotation_deg']:.1f}deg inliers={res['inliers']} confidence={res['confidence']}")
        if not args.no_images:
            save_images(slice_rgb, patch, res, name, args.out)
        row = {k: v for k, v in res.items() if k != "H"}
        row.update({"patch": p, "found": True, "transform_2x3": res["H"].tolist()})
        results.append(row)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["patch", "found", "center_x", "center_y", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
                     "scale", "rotation_deg", "inliers", "confidence"])
        for r in results:
            if r["found"]:
                wr.writerow([r["patch"], 1, *np.round(r["center"], 1), *np.round(r["bbox"], 1),
                             round(r["scale"], 4), round(r["rotation_deg"], 2), r["inliers"], r["confidence"]])
            else:
                wr.writerow([r["patch"], 0] + [""] * 10)
    print(f"Results written to {args.out}/", file=sys.stderr)


if __name__ == "__main__":
    main()
