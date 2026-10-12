#!/usr/bin/env python3
"""
precompute_pooled_features.py

Precompute per-patch pooled DenseNet/KimiaNet features for all patch PNGs.

UPDATED BEHAVIOR:
- No resizing is applied.
- Each patch is forwarded at its original spatial size.
- Patches smaller than 32x32 are skipped.
- Because patch sizes vary, features are computed one patch at a time.

For each valid patch image:
  embeddings_dir/<basename>.pt containing a 1D tensor of shape (4*F,)
  where F = DenseNet121 classifier.in_features (typically 1024),
  and we use AdaptiveAvgPool2d((2,2)) -> flatten -> length 4*F = 4096.
"""

import os
import time
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as T
import torchvision.models as models


MIN_PATCH_SIZE = 32


# ----------------------------
# Dataset over PNG paths
# ----------------------------
class PatchPathDataset(Dataset):
    """
    Returns (fname, tensor, status) for each image.

    status is one of:
      - "ok"
      - "bad_read"
      - "too_small"

    fname is the basename (as passed in filenames list).
    """
    def __init__(self, patches_dir: str, filenames: List[str], transform: T.Compose):
        self.patches_dir = patches_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[str, Optional[torch.Tensor], str]:
        fname = self.filenames[idx]
        path = os.path.join(self.patches_dir, fname)

        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return fname, None, "bad_read"

        w, h = img.size
        if w < MIN_PATCH_SIZE or h < MIN_PATCH_SIZE:
            return fname, None, "too_small"

        x = self.transform(img)
        return fname, x, "ok"


def collate_keep_list(batch):
    """
    Keep items as a list because tensors may have different H x W.
    """
    return batch


# ----------------------------
# KimiaNet loader utilities
# ----------------------------
def _unwrap_state_dict(ckpt_obj):
    """Handle common checkpoint wrappers."""
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
        return ckpt_obj
    raise ValueError(f"Checkpoint is not a dict/state_dict. Got type={type(ckpt_obj)}")


def _strip_prefix(s: str, prefix: str) -> str:
    return s[len(prefix):] if s.startswith(prefix) else s


def _make_features_state_dict_from_kimianet(sd_raw):
    """
    Convert multiple KimiaNet checkpoint key styles into torchvision DenseNet121 'features.*' keys.
    """
    sd = dict(sd_raw)

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {_strip_prefix(k, "module."): v for k, v in sd.items()}

    out = {}
    for k, v in sd.items():
        if k.startswith("features."):
            out[k] = v
        elif k.startswith("model.0."):
            kk = _strip_prefix(k, "model.0.")
            out[f"features.{kk}"] = v
        elif k.startswith("model.features."):
            kk = _strip_prefix(k, "model.")
            out[kk] = v
        else:
            out[f"features.{k}"] = v
    return out


def load_kimianet_densenet121(
    kimianet_path: str,
    device: str = "cpu",
    verbose: bool = True,
) -> models.DenseNet:
    """
    Instantiate torchvision DenseNet121 (weights=None) and load KimiaNet weights into encoder features.*.
    """
    base_model = models.densenet121(weights=None)

    ckpt = torch.load(kimianet_path, map_location=torch.device(device))
    sd_raw = _unwrap_state_dict(ckpt)
    sd_feat = _make_features_state_dict_from_kimianet(sd_raw)

    model_sd = base_model.state_dict()
    feat_keys = [k for k in model_sd.keys() if k.startswith("features.")]

    filtered = {k: v for k, v in sd_feat.items() if (k in model_sd and model_sd[k].shape == v.shape)}
    res = base_model.load_state_dict(filtered, strict=False)

    if verbose:
        matched = len(set(feat_keys).intersection(filtered.keys()))
        print("=== KimiaNet load diagnostics (precompute) ===")
        print(f"Checkpoint: {kimianet_path}")
        print(f"Filtered keys loaded: {len(filtered)}")
        print(f"Matched features.* keys: {matched}/{len(feat_keys)} ({matched/max(1,len(feat_keys)):.1%})")
        print(f"Missing keys: {len(res.missing_keys)} (examples: {res.missing_keys[:5]})")
        print(f"Unexpected keys: {len(res.unexpected_keys)}")

    return base_model


# ----------------------------
# Helpers
# ----------------------------
def list_pngs_in_dir(patches_dir: str) -> List[str]:
    fnames = [f for f in os.listdir(patches_dir) if f.lower().endswith(".png")]
    fnames.sort()
    return fnames


def out_path_for_fname(embeddings_dir: str, fname: str) -> str:
    out_name = os.path.splitext(fname)[0] + ".pt"
    return os.path.join(embeddings_dir, out_name)


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    if hrs > 0:
        return f"{hrs}h {mins}m {secs}s"
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches_dir", default="/projects/e32998/patches_benign_split", help="Folder with PNG patches")
    ap.add_argument("--embeddings_dir", default="/projects/e32998/patches_varsize_pooled4096", help="Where to save .pt features")
    ap.add_argument("--kimianet_ckpt", default="KimiaNet.pth", help="KimiaNet checkpoint path")
    ap.add_argument("--batch_size", type=int, default=128, help="Used only for loading items; forward pass is per-patch")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16", help="Storage dtype for saved features")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt files")
    ap.add_argument("--log_every", type=int, default=5, help="Print progress every N loader steps")
    args = ap.parse_args()

    os.makedirs(args.embeddings_dir, exist_ok=True)

    print(f"[INFO] patches_dir:      {args.patches_dir}")
    print(f"[INFO] embeddings_dir:   {args.embeddings_dir}")
    print(f"[INFO] kimianet_ckpt:    {args.kimianet_ckpt}")
    print(f"[INFO] device:           {args.device}")
    print(f"[INFO] batch_size:       {args.batch_size}")
    print(f"[INFO] num_workers:      {args.num_workers}")
    print(f"[INFO] overwrite:        {args.overwrite}")
    print(f"[INFO] dtype:            {args.dtype}")
    print(f"[INFO] min_patch_size:   {MIN_PATCH_SIZE}x{MIN_PATCH_SIZE}")
    print(f"[INFO] log_every:        {args.log_every}")
    print("[INFO] resizing:         disabled")
    print("[INFO] forward pass:     one patch at a time")

    # No resizing
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    filenames = list_pngs_in_dir(args.patches_dir)
    total_png = len(filenames)
    print(f"[INFO] Found {total_png} PNG files.")

    # Load KimiaNet DenseNet encoder
    base_model = load_kimianet_densenet121(args.kimianet_ckpt, device="cpu", verbose=True)
    features = base_model.features
    features.eval()
    for p in features.parameters():
        p.requires_grad = False

    pool = nn.AdaptiveAvgPool2d((2, 2))
    pool.eval()

    features = features.to(args.device)
    pool = pool.to(args.device)

    ds = PatchPathDataset(args.patches_dir, filenames, transform)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_keep_list,
        persistent_workers=(args.num_workers > 0),
    )

    save_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    seen_total = 0
    saved_total = 0
    skipped_existing = 0
    skipped_bad = 0
    skipped_too_small = 0

    start_time = time.time()

    with torch.inference_mode():
        for step, batch in enumerate(dl, start=1):
            for fname, x, status in batch:
                seen_total += 1

                if status == "bad_read":
                    skipped_bad += 1
                    continue

                if status == "too_small":
                    skipped_too_small += 1
                    continue

                out_path = out_path_for_fname(args.embeddings_dir, fname)
                if (not args.overwrite) and os.path.exists(out_path):
                    skipped_existing += 1
                    continue

                x = x.unsqueeze(0).to(args.device, non_blocking=True)

                feats = features(x)             # (1, F, h, w)
                pooled = pool(feats).flatten(1) # (1, 4*F)
                pooled = pooled.to(dtype=save_dtype).cpu()

                torch.save(pooled[0], out_path)
                saved_total += 1

            if args.log_every > 0 and (step % args.log_every == 0 or seen_total >= total_png):
                elapsed = time.time() - start_time
                rate = seen_total / elapsed if elapsed > 0 else 0.0
                remaining = (total_png - seen_total) / rate if rate > 0 else 0.0
                eta_str = format_eta(remaining)

                print(
                    f"[PROGRESS] step={step} seen={seen_total}/{total_png} "
                    f"saved={saved_total} skipped_existing={skipped_existing} "
                    f"skipped_bad={skipped_bad} skipped_too_small={skipped_too_small} "
                    f"rate={rate:.2f} patches/sec ETA={eta_str}"
                )

    total_elapsed = time.time() - start_time

    print("[DONE]")
    print(f"[SUMMARY] total_png={total_png}")
    print(f"[SUMMARY] saved={saved_total}")
    print(f"[SUMMARY] skipped_existing={skipped_existing}")
    print(f"[SUMMARY] skipped_bad={skipped_bad}")
    print(f"[SUMMARY] skipped_too_small={skipped_too_small}")
    print(f"[SUMMARY] embeddings_dir={args.embeddings_dir}")
    print(f"[SUMMARY] total_time={format_eta(total_elapsed)}")
    print("Note: saved tensors are length 4096 (DenseNet121 + AdaptiveAvgPool2d((2,2))).")
    print("Note: this version is typically slower than the resized batched version because")
    print("      variable-size patches are forwarded one at a time instead of as a stacked batch.")


if __name__ == "__main__":
    main()