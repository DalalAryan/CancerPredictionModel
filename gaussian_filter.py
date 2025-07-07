"""
Implementation of gaussian filter algorithm
"""

from itertools import product
import os
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey, imwrite
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros


def gen_gaussian_kernel(k_size, sigma):
    center = k_size // 2
    x, y = mgrid[0 - center : k_size - center, 0 - center : k_size - center]
    g = 1 / (2 * pi * sigma) * exp(-(square(x) + square(y)) / (2 * square(sigma)))
    return g


def gaussian_filter(image, k_size, sigma):
    height, width = image.shape[0], image.shape[1]
    dst_height = height - k_size + 1
    dst_width = width - k_size + 1

    image_array = zeros((dst_height * dst_width, k_size * k_size))
    for row, (i, j) in enumerate(product(range(dst_height), range(dst_width))):
        window = ravel(image[i : i + k_size, j : j + k_size])
        image_array[row, :] = window

    gaussian_kernel = gen_gaussian_kernel(k_size, sigma)
    filter_array = ravel(gaussian_kernel)

    dst = dot(image_array, filter_array).reshape(dst_height, dst_width).astype(uint8)

    return dst


if __name__ == "__main__":
    # Path to input and output folders
    input_folder = "Processed_Images"
    output_folder = "Gaussian_Images"
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            img = imread(img_path)
            gray = cvtColor(img, COLOR_BGR2GRAY)

            # Apply Gaussian filter
            gaussian3x3 = gaussian_filter(gray, 3, sigma=1)
            gaussian5x5 = gaussian_filter(gray, 5, sigma=0.8)

            # Save result images
            output_path_3x3 = os.path.join(output_folder, f"gaussian_3x3_{filename}")
            output_path_5x5 = os.path.join(output_folder, f"gaussian_5x5_{filename}")
            imwrite(output_path_3x3, gaussian3x3)
            imwrite(output_path_5x5, gaussian5x5)

            print(f"Processed and saved: {output_path_3x3}, {output_path_5x5}")