"""Recalculate patch coordinates without extracting or saving patch images.

The patch-placement functions below are copied unchanged from
patching_code_Fall2025.ipynb (identical to the original notebook).

preserves the original patch-placement functions exactly, including 
the contour smoothing, normal calculations, square construction, 
and overlap filtering. It also uses the same Otsu thresholding from 
updated notebook. 
"""

from pathlib import Path
import csv
import cv2
import numpy as np
from shapely.geometry import Polygon
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d

# CHANGE THESE FOR EACH SLICE!!
CASE = 36
MATCH = 1
STAIN = "h&e"  

BASE_FOLDER = Path("/home/wtq4672/Presentation2_base_code")
OUTPUT_FOLDER = BASE_FOLDER / "patch_visualizations"
SLICE_NAME = f"case_{CASE}_match_{MATCH}_{STAIN}"
SLICE_FOLDER = BASE_FOLDER / "full_slices" / f"case_{CASE}"
FULL_SLICE_PATH = SLICE_FOLDER / f"{SLICE_NAME}.tif"
MASK_PATH = SLICE_FOLDER / f"{SLICE_NAME}.png"
COORDINATES_PATH = OUTPUT_FOLDER / f"{SLICE_NAME}_patch_coordinates.csv"

# Original notebook parameters (do not change if matching its patch placement).
extension_factor = 1.1
step_size = 2
max_width = 2048
overlap_thresh = 0.3
weight_epithelium_coverage = 0.8
weight_background_pixels = 0.2
sigma_factor = 20


def estimate_curvature_complexity(points):
    """
    Estimate complexity of a contour based on curvature variations.
    """

    if len(points)<5:
        return 1
    
    diffs=np.gradient(points, axis=0)
    curvature=np.linalg.norm(np.gradient(diffs,axis=0),axis=1)

    # Count number of local maxima in curvature (sharp points)
    peaks=argrelextrema(curvature, np.greater)[0]
    
    #Return peaks/points for rough approx of sharp changes
    return len(peaks)/len(points)

def find_external_contours(binary_image):
    '''find all ext`ernal contours of an image'''

    contours,hierarchy=cv2.findContours(binary_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #This can be used to alter the order contour points are processed in while maintaining order
        #Think [1,2,3,4,5]->[4,5,1,2,3]
        #Set to 0 for no effect, check np documentation for more info
    return [np.roll(contour,0,axis=0) for contour in contours if len(contour)>1]


def gaussian_normals(contours):
    #Initialize list for final normals
    all_normals=[]
    smooth=[]

    #Loop through all contours
    for contour in contours:
        #Reshape contour into point pairings
        points=contour.reshape(-1,2).astype(np.float32)
        
        #Define sigma paramter(std for Gaussian Kernel from contour length)
        #Higher value of sigma means more aggressive smoothing
        #sigma,smoothed=estimate_sigma(points,1,1e-5,60)
        sigma=max(len(contour)*estimate_curvature_complexity(points)//sigma_factor,1)
        smoothed=gaussian_filter1d(points,sigma,axis=0, truncate = 100)
        print(f'Sigma of : {sigma}')
        #Apply Gaussian smoothing to X and Y values, attempting to remove noise from the contour
        #smoothed=np.apply_along_axis(lambda x: gaussian_filter1d(x,sigma,mode='reflect'),axis=0,arr=points)
        #Calculate gradients of X and Y for use then finding the tangent
        gradients=np.gradient(smoothed,axis=0)
        #Find vector norms of tangent vectors
        all_norms=np.linalg.norm(gradients,axis=1,keepdims=True)
        #Divide tangents by norms with numerical stability
        tangent_unit=gradients/(all_norms+1e-8)
        #Initialize normals
        normals=np.zeros_like(tangent_unit)
        
        normals = np.column_stack((-tangent_unit[:, 1], tangent_unit[:, 0]))
        all_normals.append(normals)
        smooth.append(smoothed.reshape(-1,1,2).astype(np.int32))

    return all_normals,smooth


def calculate_patch_corners(point, normal, binary_image, extension_factor=1.0, step_size=4, max_length=2048):
    """
    Calculate the corners of a square patch using epithelium width along the normal.

    Args:
        point (np.ndarray): The point on the contour.
        normal (np.ndarray): The normal vector at the point.
        binary_image (np.ndarray): Binary mask of the tissue.
        extension_factor (float): Factor to slightly enlarge the patch (e.g., 1.1 for 10% increase).
        step_size (int): Step size for checking tissue boundaries along the normal.
        max_length (int): Maximum distance to search for epithelium width.

    Returns:
        np.ndarray: The four corners of the square patch.
        int: The computed patch height.
    """
    height, width = binary_image.shape
    point = point.astype(int)

    # generating forward and backward points along the normal
    steps = np.arange(0, max_length, step_size)[:, None]
    points = np.clip(point - steps * normal, 0, [width - 1, height - 1]).astype(int)

    # extracting pixel values along the normal path
    values = binary_image[points[:, 1], points[:, 0]]

    # finding the first background pixel (where value == 0)
    stop = np.where(values == 0)[0]

    # computing distances
    width = stop[0] * step_size if stop.size > 0 else max_length

    # computing total epithelium width and adjust for stroma extension
    extended_length = int(width * extension_factor)
    half_size = extended_length // 2 

    # computing tangent vector (perpendicular to normal)
    tangent = np.array([-normal[1], normal[0]])

    # defining square corners (same logic as `calculate_square_corners`)
    corners = np.array([
        point + (tangent * half_size),
        point - (tangent * half_size),
        point - (tangent * half_size) - (normal * extended_length),
        point + (tangent * half_size) - (normal * extended_length)
    ])

    return corners, extended_length


def calculate_square_overlap(square1,square2):
    '''Calculate overlap ratio between squares'''

    square1=Polygon(square1)
    square2=Polygon(square2)

    if not square1.intersects(square2):
        return 0
    
    intersection_area=square1.intersection(square2).area
    smallest_area=min(square1.area,square2.area)
    return intersection_area/(smallest_area+1e-8)



def create_smarter_squares(contours,normals,binary_image,overlap_thresh):
    '''Create squares that meet overlap thresh with last square'''
    return_squares=[]
    return_patch_length = []
    
    #Loop over all contours and get corresponding normals to point
    for contour, contour_normals in zip(contours,normals):
        points=contour.reshape(-1,2)
        squares=[]
        patch_length = []
        last_square=None

        #Go over points, evaluating if next square's corner falls outside of range of last square
        for i,point in enumerate(points):

            corners,length=calculate_patch_corners(point,contour_normals[i],binary_image)
            overlap_thresh_size_fn = min(0.8,overlap_thresh + max((length - 100)*1.5*overlap_thresh/900,0))
            
            #Prevent poorly sized squares
            #Finding rough area approximation in terms of patch length
            side_length = np.sqrt(cv2.contourArea(contour))
            #Approximately: Want squares less than 50% of contour area, and minimum of 2% area
            if length<side_length*0.02:
                continue
            #Check to see if first square and check agaisnt last square to prevent long loop
            elif last_square is not None and calculate_square_overlap(last_square,corners)>overlap_thresh_size_fn:
                if Polygon(last_square).area>Polygon(corners).area and calculate_square_overlap(last_square,corners)>.98:
                    squares[-1]=corners
                    last_square=corners
                    continue
                continue

            #Check agasint all squares, prevents issues of larger patches covering thin sections of epithelium next to large ones 
            elif any(calculate_square_overlap(corners, square)>overlap_thresh_size_fn for square in squares)==True:
                continue
        
            squares.append(corners)
            patch_length.append(length)
            last_square=corners
        return_squares.append(squares)
        return_patch_length.append(patch_length)
    return return_squares, return_patch_length

def draw_squares(image,squares_list,color=[255,0,0],thickness=20):
    '''Draw squares on image'''

    result=image.copy()

    for squares in squares_list:
        for square in squares:
            points=square.astype(np.int32)
            cv2.polylines(result,[points],isClosed=True,color=color,thickness=thickness)
        
    return result

# Function to draw patches
def draw_patches(image, patch_list, color = [255,0,0], thickness = 20):
    patches_on_slice = image.copy()
    for squares in patch_list:
        for square in squares:
            points=square.astype(np.int32)
            cv2.polylines(patches_on_slice,[points],isClosed=True,color=color,thickness=thickness)
    return patches_on_slice



def process_contours(image,binary_image,overlap_thresh):
    """
    Main function to process multiple contours and generate aligned squares along them.
    
    Args:
        image (np.ndarray): Original image
        binary_image (np.ndarray): Binary mask of the shapes
        overlap_thresh (float): Maximum allowed overlap between squares        
    Returns:
        tuple: (processed image, contours, squares_list)
    """
    contours=find_external_contours(binary_image)
    if not contours:
        return TypeError('No contours found')
    
    #Condition check for square creation
    #normals=calculate_contour_normals(contours,smoothing_size)
    normals,smooth=gaussian_normals(contours)
    #normals,smooth=fourier_normals(contours,0.999)
    final_squares_list, patch_length_list = create_smarter_squares(contours,normals,binary_image,overlap_thresh=overlap_thresh)
    tissue_slice = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    
    result = draw_squares(tissue_slice,final_squares_list)
    #Visualzie smoothed contour

    try:
        cv2.drawContours(result,smooth,-1,(0,255,0),30)
    except:
        print('No smoothed contour to show')
    
    return result, contours, final_squares_list, normals, patch_length_list


def main():
    if not MASK_PATH.is_file():
        raise FileNotFoundError(f"Missing annotation mask: {MASK_PATH}")
    if not FULL_SLICE_PATH.is_file():
        raise FileNotFoundError(f"Missing full slice: {FULL_SLICE_PATH}")

    mask = cv2.imread(str(MASK_PATH))
    full_slice = cv2.imread(str(FULL_SLICE_PATH))
    if mask is None or full_slice is None:
        raise ValueError("Could not decode the mask or full slice")
    if mask.shape[:2] != full_slice.shape[:2]:
        raise ValueError(f"Mask/slice dimensions differ: {mask.shape[:2]} vs {full_slice.shape[:2]}")

    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Identical patch placement algorithm. No patch extraction or image saving.
    _, contours, final_squares, normals, patch_lengths = process_contours(
        str(FULL_SLICE_PATH), binary, overlap_thresh
    )

    squares = [np.asarray(square) for group in final_squares for square in group]
    if not squares:
        raise ValueError("No patches generated; no coordinates were saved")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    if COORDINATES_PATH.exists():
        raise FileExistsError(
            f"Coordinates file already exists: {COORDINATES_PATH}\n"
            "Rename or move it before rerunning to avoid overwriting your validated result."
        )
    with COORDINATES_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patch_index", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"])
        for index, square in enumerate(squares):
            writer.writerow([index, *square.astype(int).flatten().tolist()])

    print(f"Saved {len(squares)} patch coordinates to {COORDINATES_PATH}")


if __name__ == "__main__":
    main()
