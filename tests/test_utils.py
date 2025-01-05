import pytest
import os
import tempfile
from src import process_document

def test_process_document_success(document_files):
    for doc_path in document_files:
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            # Test processing
            result = process_document(doc_path, output_path)
            assert result == True
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            
        finally:
            # Cleanup
            if os.path.exists(output_path):
                os.remove(output_path)

def test_process_document_failure():
    result = process_document("nonexistent_file.jpg", "output.pdf")
    assert result == False