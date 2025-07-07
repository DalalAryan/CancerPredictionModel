#GrabCut mask, morphological cleanup, postprocessing, and contour extraction
# Tumor border detection using GrabCut, denoising, morphological cleanup, and contour extraction

import cv2
import numpy as np
import os

# Input and output folders
image_folder = "Processed_Images"
output_folder = "Contoured_Images_GrabCut"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(image_folder):
    if filename.lower().endswith(".png"):
        print(f"Processing file: {filename}")
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)

        # --- Denoising step (Non-Local Means, preserves edges) ---
        img_denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # --- Initialize mask for GrabCut ---
        mask = np.zeros(img.shape[:2], np.uint8)

        # --- Define a rectangle that covers most of the image (tune as needed) ---
        height, width = img.shape[:2]
        rect = (int(width*0.05), int(height*0.05), int(width*0.9), int(height*0.9))

        # --- Models used by GrabCut internally ---
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)

        # --- Run GrabCut segmentation ---
        cv2.grabCut(img_denoised, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # --- Convert GrabCut mask to binary mask (1: foreground, 0: background) ---
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

        # --- Morphological operations to clean up the mask ---
        kernel = np.ones((5, 5), np.uint8)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

        # --- Apply mask to original image to extract foreground (tumor) ---
        img_fg = img * mask2[:,:,np.newaxis]

        # --- Convert to grayscale and threshold to get binary image for contours ---
        gray = cv2.cvtColor(img_fg, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # --- Find contours in the binary mask ---
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_contour = img.copy()

        # --- Draw only the largest contour (assumed to be the tumor) ---
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(img_contour, [largest_contour], -1, (0,255,0), 2)
        else:
            print(f"No contours found in {filename}")

        # --- Save the result ---
        cv2.imwrite(os.path.join(output_folder, filename), img_contour)