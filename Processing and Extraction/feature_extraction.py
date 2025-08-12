import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from scipy.stats import skew, kurtosis, entropy
from scipy.spatial.distance import euclidean
from math import pi

# --- Feature Extraction Function ---
def extract_features_color_intensity(rgb_image, mask_image):
    if not (rgb_image.shape[:2] == mask_image.shape[:2]):
        raise ValueError("Image and mask must have the same dimensions.")
    boolean_mask = mask_image == 255
    lesion_pixels_rgb = rgb_image[boolean_mask]
    if lesion_pixels_rgb.size == 0:
        return {'error': 'No pixels found in the mask.'}
    lesion_pixels_gray = np.dot(lesion_pixels_rgb[...,:3], [0.299, 0.587, 0.114])
    lesion_pixels_hsv = cv2.cvtColor(
        lesion_pixels_rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV
    ).reshape(-1, 3)
    features = {
        'mean_R': np.mean(lesion_pixels_rgb[:, 0]),
        'mean_G': np.mean(lesion_pixels_rgb[:, 1]),
        'mean_B': np.mean(lesion_pixels_rgb[:, 2]),
        'mean_H': np.mean(lesion_pixels_hsv[:, 0]),
        'mean_S': np.mean(lesion_pixels_hsv[:, 1]),
        'mean_V': np.mean(lesion_pixels_hsv[:, 2]),
        'std_R': np.std(lesion_pixels_rgb[:, 0]),
        'std_G': np.std(lesion_pixels_rgb[:, 1]),
        'std_B': np.std(lesion_pixels_rgb[:, 2]),
        'std_H': np.std(lesion_pixels_hsv[:, 0]),
        'std_S': np.std(lesion_pixels_hsv[:, 1]),
        'std_V': np.std(lesion_pixels_hsv[:, 2]),
        'min_intensity': np.min(lesion_pixels_gray),
        'max_intensity': np.max(lesion_pixels_gray),
        'median_intensity': np.median(lesion_pixels_gray),
        'skewness': skew(lesion_pixels_gray) if len(lesion_pixels_gray) > 1 else 0.0,
        'kurtosis': kurtosis(lesion_pixels_gray) if len(lesion_pixels_gray) > 1 else 0.0,
        'entropy': entropy(np.histogram(lesion_pixels_gray, bins=256, range=(0, 256))[0][
            np.histogram(lesion_pixels_gray, bins=256, range=(0, 256))[0] > 0])
    }
    regions = regionprops(label(boolean_mask))
    if regions:
        region = max(regions, key=lambda r: r.area)
        centroid = region.centroid
        orientation = region.orientation
        y_coords, x_coords = np.where(boolean_mask)
        h1_idx = (y_coords - centroid[0]) * np.cos(orientation) - \
                 (x_coords - centroid[1]) * np.sin(orientation) > 0
        h2_idx = np.logical_not(h1_idx)
        h1_pixels = lesion_pixels_rgb[h1_idx]
        h2_pixels = lesion_pixels_rgb[h2_idx]
        if h1_pixels.size > 0 and h2_pixels.size > 0:
            features['color_asymmetry_rgb'] = euclidean(np.mean(h1_pixels, axis=0), np.mean(h2_pixels, axis=0))
        else:
            features['color_asymmetry_rgb'] = 0.0
    else:
        features['color_asymmetry_rgb'] = 0.0
    return features

# --- Main Feature Extraction ---
masks_dir = "HAM10000_segmentations_lesion"
masked_images_dir = "HAM10000_masked_output"
original_image_dirs = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
metadata_path = "HAM10000_metadata.csv"
output_csv = "ham10000_segmentation_analysis.csv"

metadata_df = pd.read_csv(metadata_path)
results = []

for filename in os.listdir(masks_dir):
    if filename.endswith("_segmentation.png"):
        print(f"Processing {filename}...")
        base_filename = filename.replace("_segmentation.png", "")
        mask_path = os.path.join(masks_dir, filename)
        masked_path = os.path.join(masked_images_dir, f"{base_filename}_masked.png")

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠️ Skipping {filename}: failed to load mask.")
            continue

        # Load original image from either part_1 or part_2
        original_bgr = None
        for folder in original_image_dirs:
            image_path = os.path.join(folder, f"{base_filename}.jpg")
            if os.path.exists(image_path):
                original_bgr = cv2.imread(image_path)
                if original_bgr is not None:
                    break
        if original_bgr is None:
            print(f"⚠️ Skipping {filename}: original image not found in part_1 or part_2.")
            continue

        # Load masked output
        masked_bgr = cv2.imread(masked_path)
        if masked_bgr is None:
            print(f"⚠️ Skipping {filename}: masked output not found.")
            continue

        # Resize for consistency
        new_size = (mask.shape[1] // 10, mask.shape[0] // 10)
        mask_resized = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
        _, binary_mask = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        original_resized = cv2.resize(original_rgb, new_size, interpolation=cv2.INTER_AREA)

        masked_gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
        _, masked_binary = cv2.threshold(masked_gray, 10, 255, cv2.THRESH_BINARY)
        masked_binary_resized = cv2.resize(masked_binary, new_size, interpolation=cv2.INTER_NEAREST)

        # Shape Features
        area = np.count_nonzero(binary_mask == 255)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours) if contours else 0
        labeled = label(binary_mask)
        regions = regionprops(labeled)
        longest_diameter = regions[0].major_axis_length if regions else 0
        shortest_diameter = regions[0].minor_axis_length if regions else 0
        circularity = (4 * pi * area / (perimeter ** 2)) if perimeter != 0 else 0
        ira = (perimeter / area) if area != 0 else 0
        irb = (perimeter / longest_diameter) if longest_diameter != 0 else 0
        irc = perimeter * ((1 / shortest_diameter - 1 / longest_diameter)
                           if shortest_diameter != 0 and longest_diameter != 0 else 0)
        ird = longest_diameter - shortest_diameter
        compactness = (perimeter ** 2) / area if area != 0 else 0

        # Color Features
        color_features = extract_features_color_intensity(original_resized, masked_binary_resized)

        # Metadata
        image_id = base_filename
        meta_row = metadata_df[metadata_df['image_id'] == image_id]
        if not meta_row.empty:
            row = meta_row.iloc[0]
            metadata = {
                'age': row['age'],
                'sex': row['sex'],
                'localization': row['localization'],
                'dx': row['dx'],
                'dx_type': row['dx_type'],
                'dataset': row['dataset'],
                'lesion_id': row['lesion_id']
            }
        else:
            metadata = {
                'age': None,
                'sex': None,
                'localization': None,
                'dx': None,
                'dx_type': None,
                'dataset': None,
                'lesion_id': None
            }

        # Combine all features
        row_data = {
            'Filename': base_filename,
            'Area (pixels)': area,
            'Perimeter (pixels)': perimeter,
            'Longest Diameter (pixels)': longest_diameter,
            'Shortest Diameter (pixels)': shortest_diameter,
            'Circularity Index': circularity,
            'IRA': ira,
            'IRB': irb,
            'IRC': irc,
            'IRD': ird,
            'Compactness': compactness,
            'image_id': image_id
        }
        row_data.update(metadata)
        if 'error' in color_features:
            print(f"⚠️ Color feature error in {filename}: {color_features['error']}")
            for key in ['mean_R', 'mean_G', 'mean_B', 'mean_H', 'mean_S', 'mean_V',
                        'std_R', 'std_G', 'std_B', 'std_H', 'std_S', 'std_V',
                        'min_intensity', 'max_intensity', 'median_intensity',
                        'skewness', 'kurtosis', 'entropy', 'color_asymmetry_rgb']:
                row_data[key] = np.nan
        else:
            row_data.update(color_features)

        results.append(row_data)

# Save results
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print("\n✅ Feature extraction complete.")
print(f"💾 Saved to {output_csv}")
print(df.head())
