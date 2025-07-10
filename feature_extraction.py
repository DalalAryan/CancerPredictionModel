import os
import cv2 as cv
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

image_folder = "Contoured_Gaussian"

images = []
for filename in os.listdir(image_folder)[:50]:
    print(f"Processing file: {filename}")
    if filename.endswith(".jpg"):
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path) 
        images.append(img)

import cv2
import numpy as np
import pandas as pd

class AnalysisData:
    def __init__(self, image, contour, ground_truth_csv):
        """
        Initialize the AnalysisData class and calculate features.
        
        :param image: The input image.
        :param contour: The contour of the object in the image.
        :param ground_truth_csv: Path to the ISIC2018_Task3_Test_GroundTruth.csv file.
        """
        self.image = image
        self.contour = contour
        self.ground_truth_csv = ground_truth_csv
        self.type = self._get_type_from_csv()

        # Calculate and assign features
        self.area = self.calculate_area()
        self.perimeter = self.calculate_perimeter()
        self.greatest_diameter = self.calculate_greatest_diameter()
        self.shortest_diameter = self.calculate_shortest_diameter()

    def calculate_area(self):
        """
        Calculate the area inside the contour in pixels.
        """
        return cv2.contourArea(self.contour)

    def calculate_perimeter(self):
        """
        Calculate the perimeter of the contour in pixels.
        """
        return cv2.arcLength(self.contour, True)

    def calculate_greatest_diameter(self):
        """
        Calculate the greatest diameter passing through the contour's centroid.
        """
        moments = cv2.moments(self.contour)
        if moments["m00"] == 0:
            return 0
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        distances = [np.linalg.norm(np.array([cx, cy]) - np.array(point[0])) for point in self.contour]
        return max(distances) * 2

    def calculate_shortest_diameter(self):
        """
        Calculate the shortest diameter passing through the contour's centroid.
        """
        moments = cv2.moments(self.contour)
        if moments["m00"] == 0:
            return 0
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        distances = [np.linalg.norm(np.array([cx, cy]) - np.array(point[0])) for point in self.contour]
        return min(distances) * 2

    def _get_type_from_csv(self):
        """
        Extract the 'dx' type from the ISIC2018_Task3_Test_GroundTruth.csv file using lesion_id.
        """
        df = pd.read_csv(self.ground_truth_csv)
        # Assuming the CSV has columns 'lesion_id' and 'dx'
        lesion_id = self.image.split('/')[-1].split('.')[0]  # Extract lesion_id from the image file name
        row = df[df['lesion_id'] == lesion_id]
        if not row.empty:
            return row.iloc[0]['dx']
        return None
