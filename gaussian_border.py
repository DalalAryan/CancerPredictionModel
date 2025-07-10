# Tumor border detection using GrabCut, denoising, morphological cleanup, and fallback to watershed

import cv2
import numpy as np
import os
from gaussian_filter import gaussian_filter  # Import custom Gaussian filter
from hair_removal import remove_hair_from_image  # Import custom hair removal function

# Input and output folders
image_folder = "Gaussian_Images"
output_folder = "Contoured_Gaussian"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(image_folder):
    if filename.lower().endswith(".png"):
        print(f"Processing file: {filename}")
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)

        # --- Convert image to grayscale ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Enhance contrast ---
        enhanced_gray = cv2.equalizeHist(gray)

        # --- Apply Gaussian filter ---
        gaussian_filtered = gaussian_filter(enhanced_gray, k_size=5, sigma=0.8)

        # --- Denoising step (Non-Local Means, preserves edges) ---
        img_denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # --- Initialize mask for GrabCut ---
        mask = np.zeros(img.shape[:2], np.uint8)

        # --- Define a rectangle that covers most of the image ---
        height, width = img.shape[:2]
        rect = (int(width * 0.05), int(height * 0.05), int(width * 0.9), int(height * 0.9))

        # --- Models used by GrabCut internally ---
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # --- Run GrabCut segmentation ---
        cv2.grabCut(img_denoised, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # --- Convert GrabCut mask to binary mask (1: foreground, 0: background) ---
        mask_binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

        # --- Morphological operations to clean up the mask ---
        kernel = np.ones((7, 7), np.uint8)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)

        # --- Find contours on the binary mask ---
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_with_contours = img.copy()
        if contours:
            # Draw only the largest contour (assumed to be the tumor)
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(img_with_contours, [largest_contour], -1, (0, 255, 0), 2)
            output_path = os.path.join(output_folder, f"contoured_{filename}")
            cv2.imwrite(output_path, img_with_contours)
            print(f"Processed and saved: {output_path}")
            min_area = 500  # Minimum area to filter out small contours
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        else:
            # --- Watershed fallback ---
            print(f"No contours found in {filename}, running watershed.")

            #Apply hair removal to image pre watershed
            img_no_hair = remove_hair_from_image(img)
            gray = cv2.cvtColor(img_no_hair, cv2.COLOR_BGR2GRAY)
            # Threshold for sure background and sure foreground
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Noise removal
            kernel_ws = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_ws, iterations=2)

            # Sure background area
            sure_bg = cv2.dilate(opening, kernel_ws, iterations=3)

            # Sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0

            img_ws = img.copy()
            markers = cv2.watershed(img_ws, markers)

            # Create a mask from watershed result
            ws_mask = np.zeros_like(gray, dtype=np.uint8)
            ws_mask[markers > 1] = 255

            # Find contours from watershed mask
            ws_contours, _ = cv2.findContours(ws_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_ws_contour = img.copy()
            if ws_contours:
                largest_ws_contour = max(ws_contours, key=cv2.contourArea)
                cv2.drawContours(img_ws_contour, [largest_ws_contour], -1, (0, 0, 255), 2)
            else:
                print(f"Watershed also failed for {filename}")

            output_path = os.path.join(output_folder, f"watershed_{filename}")
            cv2.imwrite(output_path, img_ws_contour)
            print(f"Watershed used for {filename}, saved: {output_path}")