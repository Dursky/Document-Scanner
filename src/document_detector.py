import cv2
import numpy as np
from typing import Tuple, List

class DocumentDetector:
    def __init__(self):
        self.min_area = 50000  # minimum document area in pixels

    def detect_document(self, image: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Detects a document in the image and returns its contours.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple containing bool (whether document was found) and numpy array with contours
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 75, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (descending)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
                
            # Approximate contour to polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # Check if polygon has 4 vertices (is a rectangle)
            if len(approx) == 4:
                return True, approx
        
        return False, np.array([])

    def order_points(self, points: np.ndarray) -> np.ndarray:
        """
        Orders contour points in sequence: top-left, top-right, bottom-right, bottom-left
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum of coordinates
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]  # top-left
        rect[2] = points[np.argmax(s)]  # bottom-right
        
        # Difference of coordinates
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]  # top-right
        rect[3] = points[np.argmax(diff)]  # bottom-left
        
        return rect