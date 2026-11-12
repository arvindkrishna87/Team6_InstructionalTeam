"""Draw one patch's saved coordinates on its original histology slice.

Requires a coordinates CSV generated with the matching patch-placement algorithm.
Does not extract or regenerate patch images.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Settings: change these for each visualization
# --------------------------------------------------
CASE = 32
MATCH = 1
STAIN = "h&e"  # Match the exact stain spelling in the filenames.
PATCH_INDEX = 11  # Zero-based patch index from the coordinates CSV.

BASE_FOLDER = Path("/home/wtq4672/Presentation2_base_code")
SLICE_FOLDER = BASE_FOLDER / "full_slices" / f"case_{CASE}"
OUTPUT_FOLDER = BASE_FOLDER / "patch_visualizations"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

SLICE_NAME = f"case_{CASE}_match_{MATCH}_{STAIN}"
FULL_SLICE_PATH = SLICE_FOLDER / f"{SLICE_NAME}.tif"
COORDINATES_PATH = OUTPUT_FOLDER / f"{SLICE_NAME}_patch_coordinates.csv"
OUTPUT_PATH = OUTPUT_FOLDER / f"{SLICE_NAME}_patch{PATCH_INDEX}_outlined.png"
DISPLAY_PATH = OUTPUT_FOLDER / f"{SLICE_NAME}_patch{PATCH_INDEX}_outlined_display.png"

# --------------------------------------------------
# 2. Load full slice and saved patch coordinates
# --------------------------------------------------
full_slice = cv2.imread(str(FULL_SLICE_PATH), cv2.IMREAD_COLOR)
if full_slice is None:
    raise FileNotFoundError(f"Could not load full slice: {FULL_SLICE_PATH}")

if not COORDINATES_PATH.is_file():
    raise FileNotFoundError(f"Could not find coordinates CSV: {COORDINATES_PATH}")

coordinates = pd.read_csv(COORDINATES_PATH)
patch_row = coordinates[coordinates["patch_index"] == PATCH_INDEX]
if patch_row.empty:
    raise ValueError(f"Patch index {PATCH_INDEX} was not found in {COORDINATES_PATH}")

coordinate_columns = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
points = patch_row.iloc[0][coordinate_columns].to_numpy(dtype=np.int32).reshape(4, 2)
print("Full slice dimensions:", full_slice.shape)
print("Original slice coordinates:\n", points)

# --------------------------------------------------
# 3. Validate coordinates and draw the same red outline
# --------------------------------------------------
height, width = full_slice.shape[:2]
if (
    np.any(points[:, 0] < 0)
    or np.any(points[:, 0] >= width)
    or np.any(points[:, 1] < 0)
    or np.any(points[:, 1] >= height)
):
    raise ValueError(
        "Saved coordinates fall outside the full slice. "
        "Check that the CSV and full slice belong to the same image."
    )

outlined_slice = full_slice.copy()
cv2.polylines(
    outlined_slice,
    [points.reshape((-1, 1, 2))],
    isClosed=True,
    color=(0, 0, 255),  # Red in OpenCV's BGR format.
    thickness=8,
)
if not cv2.imwrite(str(OUTPUT_PATH), outlined_slice):
    raise OSError(f"Could not save image: {OUTPUT_PATH}")
print("Outlined image saved to:", OUTPUT_PATH)

# --------------------------------------------------
# 4. Save the same presentation-style display image
# --------------------------------------------------
plt.figure(figsize=(12, 12))
plt.imshow(cv2.cvtColor(outlined_slice, cv2.COLOR_BGR2RGB))
plt.title(f"Case {CASE} - {STAIN.upper()} | Patch {PATCH_INDEX}")
plt.axis("off")
plt.tight_layout()
plt.savefig(DISPLAY_PATH, dpi=150, bbox_inches="tight")
plt.close()
print("Display image saved to:", DISPLAY_PATH)
