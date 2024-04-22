from scipy.ndimage import label, find_objects
import numpy as np
import cv2


IMAGE_SPACING_X = 0.7031
IMAGE_SPACING_Y = 0.7031
IMAGE_SPACING_Z = 2.5 



def compute_largest_diameter(binary_mask):
    
    # Label connected components in the binary mask
    labeled_array, num_features = label(binary_mask)

    # Find the objects (tumors) in the labeled array
    tumor_objects = find_objects(labeled_array)

    # Initialize the largest diameter variable
    largest_diameter = 0

    # Iterate through each tumor object
    for obj in tumor_objects:
        # Calculate the dimensions of the tumor object
        z_dim = obj[2].stop - obj[2].start
        y_dim = obj[1].stop - obj[1].start
        x_dim = obj[0].stop - obj[0].start

        # Calculate the diameter using the longest dimension
        diameter = max(z_dim * IMAGE_SPACING_Z, y_dim * IMAGE_SPACING_Y, x_dim * IMAGE_SPACING_X)

        # Update the largest diameter if necessary
        if diameter > largest_diameter:
            largest_diameter = diameter

    return largest_diameter / 10 # IN CM


def compute_shape(binary_mask):

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shape, margin = None, None

    if contours:
        tumor_contour_area = cv2.contourArea(contours[0])

        tumor_contour_perimeter = cv2.arcLength(contours[0], True)
        epsilon = 0.02 * tumor_contour_perimeter
        approx = cv2.approxPolyDP(contours[0], epsilon, True)
        num_sides = len(approx)

        # determine the shape based on the number of sides
        if num_sides < 5: shape = "Round"
        else: shape = "Irregular"

        # determine the margin characteristics based on solidity
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        tumor_solidity = tumor_contour_area / hull_area
        if tumor_solidity > 0.95: margin = "Smooth"
        elif tumor_solidity > 0.80: margin = "Irregular"
        else: margin = "Spiculated"

    return shape, margin

def generate_features(img, liver, tumor):
    
    features = {
        "lesion size (cm)": compute_largest_diameter(tumor),
        "lesion shape": "irregular",
        "lesion density (HU)": np.mean(img[tumor==1]),
        "involvement of adjacent organs:": "Yes" if np.sum(np.multiply(liver==0, tumor)) > 0 else "No"
    }
    
    return features 
