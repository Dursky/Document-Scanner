from fpdf import FPDF
import cv2
import os
import pytesseract
from pdf2image import convert_from_path
import tempfile
from PIL import Image
import io
import fitz

class PDFExporter:
    def __init__(self, output_dir='../test_data/output'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def perform_ocr(self, image):
        """Perform OCR using pytesseract and return text with positions"""
        return pytesseract.image_to_pdf_or_hocr(image, extension='pdf')

    def save_to_pdf(self, image_path, processed_image):
        # Create OCR PDF with text layer
        pdf_bytes = self.perform_ocr(processed_image)
        
        # Save temporary OCR PDF
        basename = os.path.splitext(os.path.basename(image_path))[0]
        pdf_path = os.path.join(self.output_dir, f"{basename}.pdf")
        
        with open(pdf_path, 'wb') as f:
            f.write(pdf_bytes)
            
        print(f"Saved searchable PDF with OCR: {pdf_path}")
        return pdf_path