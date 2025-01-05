import pytest
import cv2
import numpy as np
from src import DocumentDetector

def test_document_detector_initialization():
    detector = DocumentDetector()
    assert detector.min_area == 50000

def test_detect_document_success(document_files):
    detector = DocumentDetector()
    
    for doc_path in document_files:
        image = cv2.imread(doc_path)
        assert image is not None, f"Failed to load image: {doc_path}"
        
        found, corners = detector.detect_document(image)
        
        assert found == True, f"Failed to detect document in: {doc_path}"
        assert corners is not None
        assert corners.shape[1] == 2  # Should have x,y coordinates
        assert len(corners) == 4      # Should have 4 corners

def test_detect_document_failure(non_document_files):
    detector = DocumentDetector()
    
    for non_doc_path in non_document_files:
        image = cv2.imread(non_doc_path)
        assert image is not None, f"Failed to load image: {non_doc_path}"
        
        found, corners = detector.detect_document(image)
        assert found == False, f"Falsely detected document in: {non_doc_path}"

def test_order_points(document_files):
    detector = DocumentDetector()
    
    for doc_path in document_files:
        image = cv2.imread(doc_path)
        found, corners = detector.detect_document(image)
        
        if found:
            ordered = detector.order_points(corners.reshape(4, 2))
            
            # Check if points are ordered correctly
            assert ordered.shape == (4, 2)
            
            # Top-left point should have smallest sum of coordinates
            sums = ordered.sum(axis=1)
            assert np.argmin(sums) == 0