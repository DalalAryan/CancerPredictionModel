# this code actually works nowe
import os
import cv2 as cv
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from weiner_filter import WeinerFilter

## Path to Image folder
image_folder = "HAM10000_images_part_1"
#image_folder = "HAM_10000_Dataset"
## List of Images
images = []
for filename in os.listdir(image_folder)[:50]:
    print(f"Processing file: {filename}")
    if filename.endswith(".jpg"):
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path) 
        images.append(img)


## Grayscale Images
gray_scale_images = []

for image in images:
    image_np = np.array(image)  # Convert PIL Image to NumPy array
    gray_image = cv.cvtColor(image_np, cv.COLOR_BGR2GRAY) 
    # increase contrast using histogram equalization
    contrast_image = cv.equalizeHist(gray_image)
    gray_image = contrast_image
    gray_scale_images.append(gray_image)

filtered_images = []

for image in gray_scale_images:
    image_with_filter = WeinerFilter(image, (7, 7))
    output_image_with_filter = image_with_filter.estimateOutput()
    #sharpen_kernel = np.array([[0,-1,0], [-1,7,-1], [0,-1,0]])
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image_with_filter = cv.filter2D(output_image_with_filter, -1, sharpen_kernel)
    filtered_images.append(sharpened_image_with_filter)

output_folder = "Processed_Images"
os.makedirs(output_folder, exist_ok=True)

for idx, image in enumerate(gray_scale_images):
    image_with_filter = WeinerFilter(image, (7, 7))
    output_image_with_filter = image_with_filter.estimateOutput()
    #sharpen_kernel = np.array([[0,-1,0], [-1,7,-1], [0,-1,0]])
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image_with_filter = cv.filter2D(output_image_with_filter, -1, sharpen_kernel)
    sharpened_image_with_filter = np.clip(sharpened_image_with_filter, 0, 255).astype(np.uint8)
    filtered_images.append(sharpened_image_with_filter)
    print(f"Processed image with Weiner filter and sharpening: {image.shape}")

    # Save processed image
    output_path = os.path.join(output_folder, f"processed_{idx+1}.png")
    cv.imwrite(output_path, sharpened_image_with_filter)
    print(f"Saved processed image to {output_path}")