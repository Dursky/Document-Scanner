import pytest
import os
import tempfile
from src import PDFConverter
import cv2

def test_pdf_converter_initialization():
    converter = PDFConverter()
    assert converter.pdf_options is not None
    assert "layout_fun" in converter.pdf_options

def test_save_to_pdf(document_files):
    converter = PDFConverter()
    
    for doc_path in document_files:
        image = cv2.imread(doc_path)
        
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Convert and save
            converter.save_to_pdf(image, temp_path)
            
            # Check if file exists and has content
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)