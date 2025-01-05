import pytest
import os
import cv2
import numpy as np

@pytest.fixture
def test_data_dir():
    """Returns path to test data directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(current_dir), 'test_data')

@pytest.fixture
def document_files(test_data_dir):
    """Returns list of test document files."""
    doc_dir = os.path.join(test_data_dir, 'documents')
    return [os.path.join(doc_dir, f) for f in os.listdir(doc_dir) 
            if f.endswith(('.jpg', '.jpeg', '.png'))]

@pytest.fixture
def non_document_files(test_data_dir):
    """Returns list of test non-document files."""
    non_doc_dir = os.path.join(test_data_dir, 'non_documents')
    return [os.path.join(non_doc_dir, f) for f in os.listdir(non_doc_dir)
            if f.endswith(('.jpg', '.jpeg', '.png'))]

@pytest.fixture
def expected_results_files(test_data_dir):
    """Returns list of expected result files."""
    results_dir = os.path.join(test_data_dir, 'expected_results')
    return [os.path.join(results_dir, f) for f in os.listdir(results_dir)
            if f.endswith(('.jpg', '.jpeg', '.png'))]