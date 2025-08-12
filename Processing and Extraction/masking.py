import os
import cv2
import numpy as np

# Get full absolute paths for safety
root_dir = os.getcwd()
images_dirs = [
    os.path.join(root_dir, 'HAM10000_images_part_1'),
    os.path.join(root_dir, 'HAM10000_images_part_2')
]
masks_dir = os.path.join(root_dir, 'HAM10000_segmentations_lesion')
output_dir = os.path.join(root_dir, 'HAM10000_masked_output')

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# Process all masks
for filename in os.listdir(masks_dir):
    if not filename.endswith('_segmentation.png'):
        continue

    base_name = filename.replace('_segmentation.png', '')
    mask_path = os.path.join(masks_dir, filename)
    output_path = os.path.join(output_dir, f'{base_name}_masked.png')

    # Look for the matching image in both part_1 and part_2
    image_path = None
    for dir_path in images_dirs:
        possible_path = os.path.join(dir_path, f'{base_name}.jpg')
        if os.path.exists(possible_path):
            image_path = possible_path
            print(f"✅ Found image for {base_name} in: {dir_path}")
            break
        else:
            print(f"❌ Not found in: {possible_path}")

    if image_path is None:
        print(f"⚠️ Skipping {base_name}: not found in either part_1 or part_2")
        continue

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Failed to load image: {image_path}")
        continue

    # Load and resize mask if needed
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"⚠️ Failed to load mask: {mask_path}")
        continue
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Apply mask
    binary_mask = mask > 0
    binary_mask_3ch = np.stack([binary_mask]*3, axis=-1)
    masked_image = np.zeros_like(image)
    masked_image[binary_mask_3ch] = image[binary_mask_3ch]

    # Save result
    cv2.imwrite(output_path, masked_image)
    print(f"💾 Saved: {output_path}")
