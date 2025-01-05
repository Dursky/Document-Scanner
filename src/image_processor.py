import cv2
import numpy as np
from typing import Tuple, List

class ImageProcessor:
    def __init__(self):
        self.target_size = (1000, 1414)  # A4 format in proper proportion

    def perspective_transform(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Performs perspective transformation based on detected document points.
        """
        rect = points.reshape(4, 2)
        dst = np.array([
            [0, 0],
            [self.target_size[0], 0],
            [self.target_size[0], self.target_size[1]],
            [0, self.target_size[1]]
        ], dtype=np.float32)
        
        # Calculate transformation matrix
        matrix = cv2.getPerspectiveTransform(rect, dst)
        
        # Perform transformation
        warped = cv2.warpPerspective(image, matrix, self.target_size)
        return warped

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhances image contrast.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        return binary
