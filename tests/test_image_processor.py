import pytest
import cv2
import numpy as np
from src import ImageProcessor, DocumentDetector

def test_image_processor_initialization():
    processor = ImageProcessor()
    assert processor.target_size == (1000, 1414)

def test_perspective_transform(document_files):
    detector = DocumentDetector()
    processor = ImageProcessor()
    
    for doc_path in document_files:
        image = cv2.imread(doc_path)
        found, corners = detector.detect_document(image)
        
        if found:
            ordered_corners = detector.order_points(corners.reshape(4, 2))
            result = processor.perspective_transform(image, ordered_corners)
            
            assert result.shape[:2] == processor.target_size
            assert isinstance(result, np.ndarray)

def test_enhance_contrast(document_files, expected_results_files):
    processor = ImageProcessor()
    
    for doc_path, expected_path in zip(document_files, expected_results_files):
        image = cv2.imread(doc_path)
        result = processor.enhance_contrast(image)
        
        expected = cv2.imread(expected_path, cv2.IMREAD_GRAYSCALE)
        if expected is not None:
            # Compare histograms to check if the contrast enhancement is similar
            hist_result = cv2.calcHist([result], [0], None, [256], [0, 256])
            hist_expected = cv2.calcHist([expected], [0], None, [256], [0, 256])
            
            correlation = cv2.compareHist(hist_result, hist_expected, cv2.HISTCMP_CORREL)
            assert correlation > 0.5  # At least 50% correlation