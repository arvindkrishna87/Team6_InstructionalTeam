"""
Data utilities for MIL training

UPDATED to make patch ordering deterministic for attention visualization:

CHANGES MADE (Option B style stabilization):
1) get_all_patch_files(): now returns sorted(os.listdir(...))
   - ensures stable file enumeration order across runs / filesystems.

2) group_patches_by_slice(): after building the dict, sorts each slice's patch list
   - ensures stable ordering of patch paths within each (case_id, slice_id).

3) build_case_dict(): sorts stain_patches before appending
   - ensures stable ordering within each (case_id, stain, slice_list).

These changes DO NOT change which files belong to a slice/stain, only the order.
"""

import os
import re
import random
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any, Optional
from config import DATA_PATHS, VALID_CLASSES, SPLIT_CONFIG, GROUPED_CASES


def load_labels(csv_path: str = None) -> pd.DataFrame:
    """Load and preprocess labels CSV"""
    if csv_path is None:
        csv_path = DATA_PATHS["labels_csv"]

    labels = pd.read_csv(csv_path).drop(index=64, errors="ignore").reset_index(drop=True)
    return labels


def get_all_patch_files(patches_dir: str = None) -> List[str]:
    """
    Get list of all patch files.

    CHANGE #1:
    - Return a sorted list so ordering is deterministic.
    """
    if patches_dir is None:
        patches_dir = DATA_PATHS["patches_dir"]

    # Deterministic ordering (important for attention patch_idx -> filename mapping)
    all_files = sorted(os.listdir(patches_dir))
    return all_files


def group_patches_by_slice(all_files: List[str], root_dir: str) -> Dict[Tuple[int, str], List[str]]:
    """
    Group patches by case and slice ID
    Returns: {(case_id, slice_id): [patch_paths]}
    """
    case_slices = defaultdict(list)
    invalid_file_names = []
    flexibility_needed_counter = 0

    for filename in all_files:
        if not filename.endswith(".png"):
            continue

        # Try standard naming convention first
        match = re.match(r"case_(\d+)_([a-z]+_\d+)_", filename)
        if match:
            case_id = int(match.group(1))
            slice_id = match.group(2)
            key = (case_id, slice_id)
            case_slices[key].append(os.path.join(root_dir, filename))
            continue

        # Try without underscore between match/unmatched and number
        match = re.match(r"case_(\d+)_([a-z]+\d+)_", filename)
        if match:
            case_id = int(match.group(1))
            slice_id = match.group(2)
            # Add underscore between letters and numbers
            slice_id = re.sub(r"([A-Za-z])(\d)", r"\1_\2", slice_id)
            key = (case_id, slice_id)
            case_slices[key].append(os.path.join(root_dir, filename))
            flexibility_needed_counter += 1
            continue

        invalid_file_names.append(os.path.join(root_dir, filename))

    # CHANGE #2:
    # Ensure deterministic ordering of patch paths WITHIN each (case_id, slice_id)
    for k in list(case_slices.keys()):
        case_slices[k] = sorted(case_slices[k])

    # Print summary
    if invalid_file_names:
        print(f"Found {len(invalid_file_names)} files not following naming convention:")
        for f in invalid_file_names[:5]:  # Show first 5
            print(f"  {f}")
        if len(invalid_file_names) > 5:
            print(f"  ... and {len(invalid_file_names) - 5} more")
    else:
        print(f"All {flexibility_needed_counter} non-standard file names were handled.")

    return case_slices


def build_slice_to_class_map(patches: Dict, labels: pd.DataFrame) -> Dict[Tuple[int, str], int]:
    """Build mapping from (case_id, slice_id) to class label"""
    slice_to_class = {}

    for (case_id, slice_id), paths in patches.items():
        raw_label = labels.loc[labels["Case"] == case_id, "Class"]
        if not raw_label.empty and raw_label.item() in VALID_CLASSES:
            # Convert to binary: 1.0 -> 0 (benign), 3.0/4.0 -> 1 (high-grade)
            label = 0 if raw_label.item() == 1.0 else 1
            slice_to_class[(case_id, slice_id)] = label

    return slice_to_class


def split_by_case_stratified(slices_by_class: Dict, random_state: int = 42) -> Tuple[List, List, List]:
    """
    Split data by case to prevent leakage, maintaining class balance
    Returns: train_slices, val_slices, test_slices
    """
    # Build case -> label map and validate no mixed-label cases
    case_to_labels = defaultdict(set)
    for label, items in slices_by_class.items():
        for case_id, _ in items:
            case_to_labels[case_id].add(label)

    # Flatten to case list and aligned labels
    case_ids = []
    case_labels = []
    for cid, labs in case_to_labels.items():
        if len(labs) > 1:
            print(f"Warning: Case {cid} has mixed labels: {labs}")
        case_ids.append(cid)
        case_labels.append(next(iter(labs)))  # Take the first (should be only) label

    # Split cases with stratification
    train_ratio = SPLIT_CONFIG["train_ratio"]
    val_ratio = SPLIT_CONFIG["val_ratio"]

    # First split: train vs temp (val + test)
    case_train, case_temp, y_train, y_temp = train_test_split(
        case_ids,
        case_labels,
        test_size=(1 - train_ratio),
        stratify=case_labels,
        random_state=random_state,
    )

    # Second split: val vs test from temp
    val_size = val_ratio / (val_ratio + SPLIT_CONFIG["test_ratio"])
    case_val, case_test, _, _ = train_test_split(
        case_temp,
        y_temp,
        test_size=(1 - val_size),
        stratify=y_temp,
        random_state=random_state,
    )

    case_train = set(case_train)
    case_val = set(case_val)
    case_test = set(case_test)

    # Map case splits back to slice-level lists
    train_slices, val_slices, test_slices = [], [], []
    for label, items in slices_by_class.items():
        for case_id, slice_key in items:
            if case_id in case_train:
                train_slices.append((case_id, slice_key))
            elif case_id in case_val:
                val_slices.append((case_id, slice_key))
            elif case_id in case_test:
                test_slices.append((case_id, slice_key))
            else:
                print(f"Critical error! Case {case_id} not in any split")

    return train_slices, val_slices, test_slices


def build_case_dict(
    slice_list: List[Tuple],
    patches: Dict,
    slice_to_class: Dict
) -> Tuple[Dict, Dict]:
    """
    Build case dictionary and label map from slice list
    Returns: case_dict, label_map

    Determinism guarantees:
    - slice_list is processed in sorted order by (case_id, slice_id)
    - within each (case_id, stain, slice_id), patch paths are sorted
    - within each (case_id, stain), slices are sorted by slice_id
    """
    # Temporary structure keeps slice_id so we can sort slices deterministically
    tmp_case_dict = defaultdict(lambda: defaultdict(list))
    label_map = {}

    # IMPORTANT: deterministic ordering of slices
    slice_list_sorted = sorted(slice_list, key=lambda x: (int(x[0]), str(x[1])))

    for case_id, slice_id in slice_list_sorted:
        if (case_id, slice_id) not in patches:
            continue

        patch_paths = patches[(case_id, slice_id)]

        # Group patches by stain (extract stain from filenames)
        stain_groups = defaultdict(list)
        for patch_path in patch_paths:
            stain = extract_stain_from_filename(patch_path)  # <-- data_utils.py
            if stain:
                stain_groups[stain].append(patch_path)

        # Store (slice_id, sorted_patch_list) so slices can be sorted later
        for stain in sorted(stain_groups.keys()):
            stain_patches = sorted(stain_groups[stain])
            tmp_case_dict[case_id][stain].append((str(slice_id), stain_patches))

        # Set label for this case (stable: label is per-case anyway)
        if (case_id, slice_id) in slice_to_class:
            label_map[case_id] = slice_to_class[(case_id, slice_id)]

    # Convert tmp_case_dict -> expected structure: case_dict[case_id][stain] = List[List[str]]
    case_dict = {}
    for case_id, stain_map in tmp_case_dict.items():
        case_dict[case_id] = {}
        for stain, slice_entries in stain_map.items():
            # Deterministic ordering of slices within each stain
            slice_entries_sorted = sorted(slice_entries, key=lambda t: t[0])  # sort by slice_id
            case_dict[case_id][stain] = [patch_list for (_, patch_list) in slice_entries_sorted]

    return case_dict, label_map

def extract_stain_from_filename(filename: str) -> Optional[str]:
    """Extract stain type from patch filename"""
    filename_lower = filename.lower()
    if "h&e" in filename_lower or "_he_" in filename_lower:
        return "h&e"
    elif "melan" in filename_lower:
        return "melan"
    elif "sox10" in filename_lower:
        return "sox10"
    return None


def get_case_ids(case_dict: Dict) -> set:
    """Extract unique case IDs from case dictionary"""
    return set(case_dict.keys())


def get_all_paths(case_dict: Dict) -> set:
    """Extract all patch paths from case dictionary"""
    paths = set()
    for case_data in case_dict.values():
        for stain_data in case_data.values():
            for slice_paths in stain_data:
                paths.update(slice_paths)
    return paths


def check_disjoint_sets(set1: set, set2: set, name1: str, name2: str) -> Tuple[bool, set]:
    """Check if two sets are disjoint and return overlap"""
    overlap = set1.intersection(set2)
    return len(overlap) == 0, overlap


def report_no_leak(train_case_dict: Dict, val_case_dict: Dict, test_case_dict: Dict):
    """Report data leakage analysis"""
    # Case-level analysis
    train_cases = get_case_ids(train_case_dict)
    val_cases = get_case_ids(val_case_dict)
    test_cases = get_case_ids(test_case_dict)

    print("Cases per split:", len(train_cases), len(val_cases), len(test_cases))

    # Replace pseudo-case id with real case id for accurate leak detection
    cases_remap = {case: group[0] for group in GROUPED_CASES for case in group}
    train_cases = {cases_remap.get(item, item) for item in train_cases}
    val_cases = {cases_remap.get(item, item) for item in val_cases}
    test_cases = {cases_remap.get(item, item) for item in test_cases}

    ok_tv, leak_tv = check_disjoint_sets(train_cases, val_cases, "train", "val")
    ok_tt, leak_tt = check_disjoint_sets(train_cases, test_cases, "train", "test")
    ok_vt, leak_vt = check_disjoint_sets(val_cases, test_cases, "val", "test")

    # Path-level analysis
    train_paths = get_all_paths(train_case_dict)
    val_paths = get_all_paths(val_case_dict)
    test_paths = get_all_paths(test_case_dict)

    print("Paths per split:", len(train_paths), len(val_paths), len(test_paths))

    ok_tv_p, leak_tv_p = check_disjoint_sets(train_paths, val_paths, "train", "val")
    ok_tt_p, leak_tt_p = check_disjoint_sets(train_paths, test_paths, "train", "test")
    ok_vt_p, leak_vt_p = check_disjoint_sets(val_paths, test_paths, "val", "test")

    # Summary
    def summarise(ok, leak, label):
        if ok:
            print(f"No leakage between {label}.")
        else:
            print(f"[LEAK!!!! Nooo] {label} overlap count = {len(leak)}")

    summarise(ok_tv, leak_tv, "train & val (cases)")
    summarise(ok_tt, leak_tt, "train & test (cases)")
    summarise(ok_vt, leak_vt, "val & test (cases)")
    summarise(ok_tv_p, leak_tv_p, "train & val (paths)")
    summarise(ok_tt_p, leak_tt_p, "train & test (paths)")
    summarise(ok_vt_p, leak_vt_p, "val & test (paths)")


def summarize_case_dict(case_dict: Dict, label_map: Dict = None, split_name: str = "train") -> pd.DataFrame:
    """
    Create summary DataFrame with per-case statistics
    """
    records = []

    for case_id, stains in case_dict.items():
        record = {"case_id": case_id, "split": split_name}
        total_patches = 0

        for stain in ("h&e", "melan", "sox10"):
            slice_lists = stains.get(stain, [])
            num_slices = len(slice_lists)
            num_patches = sum(len(paths) for paths in slice_lists)
            record[f"{stain}_slices"] = num_slices
            record[f"{stain}_patches"] = num_patches
            record[f"{stain}_missing"] = int(num_patches == 0)
            total_patches += num_patches

        record["total_patches"] = total_patches
        if label_map and case_id in label_map:
            record["label"] = label_map[case_id]
        else:
            record["label"] = None

        records.append(record)

    return pd.DataFrame.from_records(records)