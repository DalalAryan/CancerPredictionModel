import cv2
import os

def apply_hair_removal(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_folder, filename)
            
            # Read image
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Warning: Could not read image {filename}. Skipping.")
                continue

            # Image cropping (adjust coordinates as needed for your dataset)
            # You might want to make this dynamic or remove if images are already pre-cropped
            # For now, let's assume a default crop if it's generally applicable
            # If not, remove this line to process the full image
            img = image[30:410, 30:560] 
            
            # Check if cropping resulted in an empty image
            if img.shape[0] == 0 or img.shape[1] == 0:
                print(f"Warning: Cropping resulted in an empty image for {filename}. Skipping.")
                continue

            # DULL RAZOR (REMOVE HAIR)

            # Gray scale
            grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Black hat filter - You might need to experiment with kernel size (e.g., (15,15) or (21,21))
            # Larger kernels detect thicker/longer hairs
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)) # Using MORPH_RECT is often better for hair
            blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
            
            # Gaussian filter
            # Adjust kernel size (3,3) for smaller blur, (5,5) or higher for more blur
            bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
            
            # Binary thresholding (MASK) - Threshold '10' might need tuning
            # This threshold determines what is considered 'hair' based on blackhat response
            ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
            
            # Replace pixels of the mask - '6' is the inpainting radius
            # Adjust inpainting radius based on hair thickness
            dst = cv2.inpaint(img, mask, 6, cv2.INPAINT_TELEA)

            # Save the clean image
            output_filename = f"hair_removed_{filename}"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, dst)
            print(f"Processed and saved: {output_filename}")

def remove_hair_from_image(img):
    # No cropping unless needed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))  # Larger kernel
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    bhg = cv2.GaussianBlur(blackhat, (5, 5), 0)
    # Try Otsu's thresholding for adaptive mask
    _, mask = cv2.threshold(bhg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dst = cv2.inpaint(img, mask, 9, cv2.INPAINT_TELEA)  # Slightly larger radius
    return dst

# Define your input and output folders
input_folder = "Gaussian_Images" # Assuming your initial images are here
output_folder = "HairRemoval_Images"

# Run the hair removal
apply_hair_removal(input_folder, output_folder)

print("Hair removal complete for all images.")

# The display part below is for individual testing, not for batch processing
# cv2.imshow("Original image",image)
# cv2.imshow("Cropped image",img)
# cv2.imshow("Gray Scale image",grayScale)
# cv2.imshow("Blackhat",blackhat)
# cv2.imshow("Binary mask",mask)
# cv2.imshow("Clean image",dst)
# cv2.waitKey()
# cv2.destroyAllWindows()