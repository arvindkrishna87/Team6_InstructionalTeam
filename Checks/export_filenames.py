#!/usr/bin/env python3
"""
export_filenames.py

Export every filename in a folder to a CSV, with the case / slice / stain /
patch index parsed out so two exports can be matched on everything except
the case number.

Usage:
    python export_filenames.py /projects/e32998/patches               patches.csv
    python export_filenames.py /projects/e32998/patches_benign_split  patches_benign_split.csv
    python export_filenames.py /projects/e32998/patches_Team5         patches_Team5.csv
"""
import os
import re
import csv
import sys


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    folder, out_csv = sys.argv[1], sys.argv[2]

    # case_<id>_<slice>_<stain>[-labels][.]_patch<n>.png
    # e.g. case_25_unmatched_4_h&e_patch35.png
    pat = re.compile(
        r"^case_(?P<case>\d+)_(?P<slice>(?:match|unmatched)_?\d+)_(?P<stain>h&e|melan|sox10)"
        r"(?:-labels)?\.?_patch(?P<patch>\d+)\.png$",
        re.IGNORECASE,
    )

    n_total = n_parsed = 0
    with open(out_csv, "w", newline="") as f, os.scandir(folder) as it:
        w = csv.writer(f)
        w.writerow(["filename", "case", "slice", "stain", "patch", "rest"])
        for e in it:
            if not e.is_file():
                continue
            n_total += 1
            m = pat.match(e.name)
            if m:
                n_parsed += 1
                rest = e.name.split("_", 2)[2]  # everything after case_<id>_
                w.writerow([
                    e.name,
                    int(m["case"]),
                    m["slice"].lower(),
                    m["stain"].lower(),
                    int(m["patch"]),
                    rest,
                ])
            else:
                w.writerow([e.name, "", "", "", "", ""])

    print(f"{folder}: {n_total} files, {n_parsed} parsed, "
          f"{n_total - n_parsed} unparsed -> {out_csv}")


if __name__ == "__main__":
    main()
