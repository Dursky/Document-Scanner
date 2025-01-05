import cv2
import numpy as np
from typing import Tuple, List
from .document_detector import DocumentDetector
from .image_processor import ImageProcessor
from .pdf_converter import PDFConverter

def load_image(path: str) -> np.ndarray:
    """
    Loads image from file.
    """
    return cv2.imread(path)

def save_image(image: np.ndarray, path: str) -> None:
    """
    Saves image to file.
    """
    cv2.imwrite(path, image)

# Example usage:
def process_document(input_path: str, output_path: str) -> bool:
    """
    Main document processing function.
    """
    # Initialize classes
    detector = DocumentDetector()
    processor = ImageProcessor()
    pdf_converter = PDFConverter()
    
    # Load image
    image = load_image(input_path)
    if image is None:
        return False
    
    # Detect document
    found, corners = detector.detect_document(image)
    if not found:
        return False
    
    # Order points
    ordered_corners = detector.order_points(corners.reshape(4, 2))
    
    # Perspective transformation
    warped = processor.perspective_transform(image, ordered_corners)
    
    # Enhance contrast
    enhanced = processor.enhance_contrast(warped)
    
    # Save to PDF
    pdf_converter.save_to_pdf(enhanced, output_path)
    
    return True