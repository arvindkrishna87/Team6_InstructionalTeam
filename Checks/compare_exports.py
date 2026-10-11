#!/usr/bin/env python3
"""
compare_exports.py

Compare CSVs produced by export_filenames.py across the three patch folders:

    patches               original tiling, untouched
    patches_Team5         after the renaming notebook (pseudo-cases 107-119 created in place)
    patches_benign_split  cp -r of patches_Team5, then badly-named files purged

Part 1  patches -> patches_Team5
        Treats each (slice, stain) as a fingerprint: finds which original case
        lost a slice, which pseudo-case (>= 107) gained it, and checks the patch
        counts agree.  This is the rename-only comparison and is the one that
        answers "which parent did case N come from?".

Part 2  patches_Team5 -> patches_benign_split
        Lists exactly which files the purge removed, grouped by case.

Usage:
    python compare_exports.py patches.csv patches_Team5.csv patches_benign_split.csv
    python compare_exports.py patches.csv patches_Team5.csv          # part 1 only
"""
import sys
import pandas as pd


def load(path):
    df = pd.read_csv(path)
    parsed = df.dropna(subset=["case"]).copy()
    parsed["case"] = parsed["case"].astype(int)
    return df, parsed


def per_slice_counts(df, name):
    return df.groupby(["case", "slice", "stain"]).size().rename(name)


def rename_report(orig, team5):
    o = per_slice_counts(orig, "n_orig")
    t = per_slice_counts(team5, "n_team5")

    lost = o[~o.index.isin(t.index)].reset_index()
    gained = t[t.index.get_level_values("case") >= 107].reset_index()

    print("=" * 70)
    print("PART 1: patches -> patches_Team5  (the renaming step)")
    print("=" * 70)

    print("\nSlices removed from original cases:")
    print(lost.to_string(index=False) if len(lost) else "  (none)")

    print("\nSlices under pseudo-cases (>=107):")
    print(gained.to_string(index=False) if len(gained) else "  (none)")

    m = lost.merge(gained, on=["slice", "stain"], suffixes=("_orig", "_team5"))
    m = m[m.n_orig == m.n_team5][["case_orig", "case_team5", "slice", "stain", "n_orig"]]
    print("\nInferred parent -> pseudo-case mapping:")
    print(m.sort_values("case_team5").to_string(index=False) if len(m) else "  (none)")

    unmatched = sorted(set(gained.case) - set(m.case_team5))
    print("\nPseudo-cases with no parent match:", unmatched)
    for c in (112, 113):
        status = "present" if c in set(gained.case) else "ABSENT (never created)"
        print(f"  case {c}: {status}")

    # sanity: a pure rename keeps the file count identical
    print(f"\nFile counts  patches: {len(orig)}   patches_Team5: {len(team5)}   "
          f"diff: {len(orig) - len(team5)}  (should be 0 for a pure rename)")

    # per-parent check: patches lost by parent == patches gained by its children
    if len(m):
        print("\nPer-parent patch accounting:")
        for parent, grp in m.groupby("case_orig"):
            n_parent_orig = int((orig.case == parent).sum())
            n_parent_now = int((team5.case == parent).sum())
            n_children = int(team5.case.isin(grp.case_team5).sum())
            ok = n_parent_orig == n_parent_now + n_children
            print(f"  case {parent:>3}: {n_parent_orig} -> {n_parent_now} + "
                  f"{n_children} (children {sorted(grp.case_team5)})  {'OK' if ok else 'MISMATCH'}")


def purge_report(team5_all, split_all):
    print("\n" + "=" * 70)
    print("PART 2: patches_Team5 -> patches_benign_split  (the purge step)")
    print("=" * 70)

    t_names = set(team5_all.filename)
    s_names = set(split_all.filename)
    removed = sorted(t_names - s_names)
    added = sorted(s_names - t_names)

    print(f"\nFile counts  patches_Team5: {len(t_names)}   patches_benign_split: {len(s_names)}")
    print(f"Removed by purge: {len(removed)}   Added (unexpected): {len(added)}")

    if removed:
        rm = team5_all[team5_all.filename.isin(removed)]
        by_case = rm.groupby(rm["case"].fillna("unparsed")).size().sort_index()
        print("\nRemoved files by case:")
        print(by_case.to_string())
        print("\nFirst 15 removed filenames:")
        for f in removed[:15]:
            print("  ", f)
        for c in (112, 113):
            n = int((rm.case == c).sum())
            if n:
                print(f"\n  NOTE: {n} files of case {c} were deleted by the purge")

    if added:
        print("\nFiles present in patches_benign_split but not in patches_Team5:")
        for f in added[:15]:
            print("  ", f)


def main():
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(1)

    orig_all, orig = load(sys.argv[1])
    team5_all, team5 = load(sys.argv[2])

    rename_report(orig, team5)

    if len(sys.argv) == 4:
        split_all, _ = load(sys.argv[3])
        purge_report(team5_all, split_all)


if __name__ == "__main__":
    main()
