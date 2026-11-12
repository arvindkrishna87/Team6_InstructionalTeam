from pathlib import Path
import csv

folders = [
    Path("/projects/e32998/STAT390_Fall2026/Team2_TheConsultancy/Presentation2/Callie_Presentation2_data/1_new_case_patches/patches"),
]
export_dir = Path("/projects/e32998/STAT390_Fall2026/Team6_InstructionalTeam/Code/Presentation3_base_code/checking_new_patches/")
for folder in folders:
    # Save each CSV alongside its folder
    output_csv = export_dir / f"{folder.name}_files.csv"

    file_names = sorted(
        path.name for path in folder.iterdir() if path.is_file()
    )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"])
        writer.writerows([name] for name in file_names)

    print(f"Exported {len(file_names)} filenames to {output_csv.resolve()}")
