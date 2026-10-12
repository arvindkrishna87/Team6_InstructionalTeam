#!/usr/bin/env python3
"""
make_splits.py

Creates a NEW train/val/test split (by CASE), prints benign vs high-grade CASE IDs
in each split, and saves a data_splits.npz compatible with:

  python main.py --load_splits /path/to/data_splits.npz

ENFORCED:
- Split by CASE (no leakage).
- Group specified cases together.
- Keep split sizes close to target ratios (default 60/20/20) and NON-EMPTY.
- Match benign/high-grade proportions across Train/Val/Test as tightly as possible.

Why not "exact" always?
Exact equality of proportions can be mathematically impossible unless split sizes
are compatible with the reduced fraction of (H_total / N_total). In those cases,
forcing exact equality can collapse val/test to 0. This script instead enforces
a near-exact rational approximation with a small denominator to keep all splits
non-empty and proportions effectively equal.

Labels:
  0 = benign (Class 1.0)
  1 = high-grade (Class 3.0 or 4.0)
"""

import os
import argparse
import numpy as np
import ast
import math
from sklearn.model_selection import StratifiedGroupKFold
from fractions import Fraction
from functools import reduce
from collections import deque
from config import DATA_PATHS, TRAINING_CONFIG, SPLIT_CONFIG, GROUPED_CASES
from data_utils import (
    load_labels,
    get_all_patch_files,
    group_patches_by_slice,
    build_slice_to_class_map,
    build_case_dict,
    report_no_leak,
)
from utils import save_data_splits


def _split_counts(case_ids, case_to_label):
    n = len(case_ids)
    h = sum(1 for c in case_ids if case_to_label[c] == 1)
    b = n - h
    return b, h, n


def _print_split(name, case_ids, case_to_label):
    benign = sorted([c for c in case_ids if case_to_label[c] == 0])
    high = sorted([c for c in case_ids if case_to_label[c] == 1])
    total = len(case_ids)
    ratio = (len(high) / total) if total else 0.0

    print("\n" + "=" * 40)
    print(f"{name.upper()} SPLIT")
    print("=" * 40)
    print(f"Total cases:      {total}")
    print(f"Benign (0):       {len(benign)}")
    print(f"High-grade (1):   {len(high)}")
    print(f"High-grade ratio: {ratio:.6f}")

    print("\nBenign CASE IDs:")
    print(benign)
    print("\nHigh-grade CASE IDs:")
    print(high)


def simplify_split_ratios(ratios, max_ratio_den):
    """Finds a suitable number of folds given ratios"""

    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.")

    best_integers = []
    min_error = float('inf')

    # Iterate through all possible total sums (denominators)
    for d in range(1, max_ratio_den + 1):
        # Initial rounding for this specific total 'd'
        ints = [round(r * d) for r in ratios]
        
        # Adjust if rounding caused the sum to not equal 'd'
        while sum(ints) != d:
            diff = d - sum(ints)
            # Find which index has the largest rounding error to nudge it
            errors = [r * d - i for r, i in zip(ratios, ints)]
            idx = np.argmax(errors) if diff > 0 else np.argmin(errors)
            ints[idx] += 1 if diff > 0 else -1

        # Calculate total error (how far are we from original ratios?)
        current_error = sum(abs(r - (i / d)) for r, i in zip(ratios, ints))
        
        if current_error < min_error:
            min_error = current_error
            best_integers = ints

    # Final pass to simplify by GCD
    common_gcd = reduce(math.gcd, best_integers)
    return [i // common_gcd for i in best_integers]


def split_by_case_with_constraints(
    slice_to_class: dict,
    grouped_cases: tuple,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    max_ratio_den: int | None = None,
):
    """
    Split by case (including grouped benign cases treated together) to avoid data leakage
    Using StratifiedGroupKFold to ensure representative label distribution and randomized splits
    """

    case_to_label = {}
    for (case_id, _slice_id), y in slice_to_class.items():
        y = int(y)
        case_to_label.setdefault(case_id, y)
    case_ids = sorted(case_to_label.keys())

    N_total = len(case_to_label)
    if N_total == 0:
        raise RuntimeError("No cases found to split.")

    H_total = sum(1 for c in case_ids if case_to_label[c] == 1)
    B_total = N_total - H_total
    if H_total == 0 or B_total == 0:
        raise RuntimeError(
            "Cannot enforce matched proportions if only one class exists.\n"
            f"Counts: benign={B_total}, high={H_total}"
        )

    print("\nOverall case counts:")
    print(f"  Total:  {N_total}  |  Benign: {B_total}  |  High: {H_total}")

    # Identify reasonable split ratios
    ratios = [train_ratio, val_ratio, test_ratio]
    denom = simplify_split_ratios(ratios, max_ratio_den)

    # Setup groups so that pseudo-cases can be mapped back to original case
    groups = np.array(case_ids)
    for group in grouped_cases:
        mask = np.isin(groups, group)
        np.place(groups, mask, group[0])
    
    # Perform split according to smallest unit that can represent one split set
    sgkf = StratifiedGroupKFold(n_splits=sum(denom)) # shuffle disabled for causing deviations of labels ratio
    sgkf_splits = sgkf.split(
        np.array(case_ids), 
        np.array([case_to_label[case_ids] for case_ids in case_ids]), 
        groups
    )

    # Folds of only "test" for non overlapping sets
    splits = []
    for _, test_index in sgkf_splits:
        splits.append([case_ids[index] for index in test_index])

    return case_to_label, denom, splits

def main():
    ap = argparse.ArgumentParser(description="Create and save train/val/test case splits for MIL training.")
    ap.add_argument("--labels_csv", type=str, default=DATA_PATHS["labels_csv"])
    ap.add_argument("--patches_dir", type=str, default=DATA_PATHS["patches_dir"])
    ap.add_argument("--seed", type=int, default=TRAINING_CONFIG["random_state"])
    ap.add_argument("--save_dir", type=str, default=".")
    ap.add_argument(
        "--grouped_cases",
        type=str,
        default=str(GROUPED_CASES),
        help="List of grouped case IDs in tuples",
    )
    ap.add_argument("--train_ratio", type=float, default=float(SPLIT_CONFIG["train_ratio"]))
    ap.add_argument("--val_ratio", type=float, default=float(SPLIT_CONFIG["val_ratio"]))
    ap.add_argument("--test_ratio", type=float, default=float(SPLIT_CONFIG["test_ratio"]))
    ap.add_argument(
        "--max_ratio_den",
        type=int,
        default=20,
        help="Max denominator for ratio approximation (smaller => easier feasibility, larger => closer to global ratio).",
    )
    args = ap.parse_args()

    grouped_cases = ast.literal_eval(args.grouped_cases)

    # Summary of input case data
    print("=" * 80)
    print("CREATING DATA SPLITS (GROUPED CASES + MATCHED PROPORTIONS)")
    print("=" * 80)
    print(f"labels_csv:        {args.labels_csv}")
    print(f"patches_dir:       {args.patches_dir}")
    print(f"seed:              {args.seed}")
    print(f"save_dir:          {args.save_dir}")
    print(f"grouped_cases:     {sorted(grouped_cases)}")
    print(f"ratios (targets):  train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"max_ratio_den:     {args.max_ratio_den}")

    labels = load_labels(args.labels_csv)
    print(f"\nLoaded {len(labels)} label rows")

    all_files = get_all_patch_files(args.patches_dir)
    print(f"Found {len(all_files)} files in patches_dir")

    patches = group_patches_by_slice(all_files, args.patches_dir)
    print(f"Grouped into {len(patches)} slices")

    slice_to_class = build_slice_to_class_map(patches, labels)
    print(f"Mapped {len(slice_to_class)} slices to classes")

    case_to_label, denom, splits = split_by_case_with_constraints(
        slice_to_class=slice_to_class,
        grouped_cases=grouped_cases,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_ratio_den=args.max_ratio_den,
    )

    # Data split specific data summary, creating data splits to test all cases
    for i in range(math.ceil(sum(denom)/denom[2])):

        # Assign folds to train/val/test
        train_cases = np.concatenate(splits[0:denom[0]])
        val_cases = np.concatenate(splits[denom[0]:denom[0]+denom[1]])
        test_cases = np.concatenate(splits[-denom[2]:])

        print("\n" + "=" * 80)
        print(f"Fold {i+1}")
        print("=" * 80)

        # Ratio printout (they will be extremely close / often identical up to rounding)
        def ratio_str(cset):
            b, h, n = _split_counts(list(cset), case_to_label)
            return f"{h}/{n} = {h/n:.6f}"

        print("\nFinal achieved high-grade ratios:")
        print(f"  Train: {ratio_str(train_cases)}")
        print(f"  Val:   {ratio_str(val_cases)}")
        print(f"  Test:  {ratio_str(test_cases)}")

        # Convert to slice-level lists
        train_slices, val_slices, test_slices = [], [], []
        for (case_id, slice_id), _y in slice_to_class.items():
            if case_id in train_cases:
                train_slices.append((case_id, slice_id))
            elif case_id in val_cases:
                val_slices.append((case_id, slice_id))
            elif case_id in test_cases:
                test_slices.append((case_id, slice_id))
            else:
                print(f"Critical error! Case {case_id} not in any split")

        print("\nSlice counts after split:")
        print(f"  Train slices: {len(train_slices)}")
        print(f"  Val slices:   {len(val_slices)}")
        print(f"  Test slices:  {len(test_slices)}")

        # Build dicts for leak checks + printing lists
        train_case_dict, train_label_map = build_case_dict(train_slices, patches, slice_to_class)
        val_case_dict, val_label_map = build_case_dict(val_slices, patches, slice_to_class)
        test_case_dict, test_label_map = build_case_dict(test_slices, patches, slice_to_class)

        print("\n" + "-" * 40)
        print("LEAK CHECK")
        print("-" * 40)
        report_no_leak(train_case_dict, val_case_dict, test_case_dict)

        # Print case lists per split
        _print_split("train", list(train_case_dict.keys()), case_to_label)
        _print_split("val", list(val_case_dict.keys()), case_to_label)
        _print_split("test", list(test_case_dict.keys()), case_to_label)

        # Save splits
        os.makedirs(args.save_dir, exist_ok=True)
        train_cases = sorted(list(train_case_dict.keys()))
        val_cases = sorted(list(val_case_dict.keys()))
        test_cases = sorted(list(test_case_dict.keys()))
        save_data_splits(
            train_cases, val_cases, test_cases, 
            save_dir=args.save_dir, name=f"data_splits_new_{i+1:02}.npz"
        )

        out_path = os.path.join(args.save_dir, f"data_splits_new_{i+1:02}.npz")
        print("\n" + "=" * 40)
        print("DONE")
        print("=" * 40)
        print(f"Saved: {out_path}")
        print("Use it like:")
        print(f"  python main.py --load_splits {out_path}")

        # rotate splits
        splits = splits[denom[2]:] + splits[:denom[2]]

if __name__ == "__main__":
    main()