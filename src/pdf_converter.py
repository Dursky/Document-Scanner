import img2pdf
from PIL import Image
import io

class PDFConverter:
    def __init__(self):
        self.pdf_options = {
            "layout_fun": img2pdf.get_layout_fun((img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297)))  # A4
        }

    def save_to_pdf(self, image: np.ndarray, output_path: str) -> None:
        """
        Saves processed image to PDF file.
        """
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Convert to PDF and save
        with open(output_path, "wb") as f:
            f.write(img2pdf.convert(pil_image, **self.pdf_options))