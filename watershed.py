import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt 
from scipy.ndimage import binary_dilation, generate_binary_structure

def apply_watershed_segmentation(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_folder, filename)
            
            img = cv.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {filename}. Skipping.")
                continue

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

            # noise removal
            kernel = np.ones((3, 3), np.uint8)
            opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

            # sure background area
            sure_bg = cv.dilate(opening, kernel, iterations=3)

            # Finding sure foreground area
            dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
            ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv.subtract(sure_bg, sure_fg)

            # Marker labelling
            ret, markers = cv.connectedComponents(sure_fg)

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1

            # Now, mark the region of unknown with zero
            markers[unknown == 255] = 0

            # Perform watershed
            markers = cv.watershed(img, markers)

            # Create a mask of the watershed boundaries
            boundary_mask = (markers == -1)

            # Define the structuring element for dilation to make the lines wider
            # You can adjust the size and shape of the structure for different thickness
            struct = generate_binary_structure(2, 2) # This creates a 2x2 cross shape

            # Dilate the boundary mask
            dilated_boundary_mask = binary_dilation(boundary_mask, structure=struct, iterations=1) # Adjust iterations for more dilation

            # Create a copy of the original image to draw boundaries on
            segmented_img = img.copy()
            segmented_img [dilated_boundary_mask] = [255, 0, 0] # Mark dilated watershed boundaries in red

            # Save the segmented image
            output_filename = f"watershed_{filename}"
            output_path = os.path.join(output_folder, output_filename)
            cv.imwrite(output_path, segmented_img)
            print(f"Processed and saved: {output_filename}")

# Define your input and output folders
input_folder = "HairRemoval_Images"
output_folder = "WatershedAfterHairRemoval_Images"

# Run the segmentation
apply_watershed_segmentation(input_folder, output_folder)

print("Watershed segmentation complete for all images.")