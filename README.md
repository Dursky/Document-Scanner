# 📄 Document-Scanner

🔍 Advanced document scanner with deep learning-based document classification and adaptive processing parameters.

## ✨ Features

- Document type classification using deep learning
- Adaptive processing parameters based on document type
- PDF export with OCR
- JSON metadata output
- Optimized for macOS

## 🔧 Installation

```bash
# Create conda environment
conda create -n docscanner python=3.11
conda activate docscanner

# Install required packages
conda install -c conda-forge pytorch torchvision
conda install -c conda-forge python-fitz PyMuPDF pytesseract
conda install fpdf2 opencv matplotlib
```

## 📁 Project Structure

```
document-scanner/
├── src/
│   ├── modules/
│   │   ├── document_classifier.py  # DL classification
│   │   ├── train_classifier.py     # Training script
│   │   ├── transform.py           # Image transformations
│   │   ├── utils.py              # Utilities
│   │   └── pdf_exporter.py       # PDF generation
│   └── scanner.py                # Main script
├── dataset/                      # Training data
└── test_data/
    ├── documents/               # Test documents
    └── output/                  # Results
```

## 🚀 Usage

### Train the classifier:

```bash
python modules/train_classifier.py --data_dir ../dataset
```

### Scan documents:

```bash
# Single document
python scanner.py --image path/to/image.jpg

# Multiple documents
python scanner.py --images path/to/directory
```

## 📚 Dataset Structure

```
dataset/
├── advertisement/
├── budget/
├── email/
├── file_folder/
├── form/
├── handwritten/
├── invoice/
├── letter/
├── memo/
├── news_article/
├── presentation/
├── questionnaire/
├── resume/
├── scientific_publication/
├── scientific_report/
└── specification/
```

## 📊 Example Training Results

```
Epoch 1/2
Training Loss: 1.863
Training Accuracy: 43.23%
Validation Loss: 1.530
Validation Accuracy: 51.90%

Epoch 2/2
Training Loss: 1.432
Training Accuracy: 56.48%
Validation Loss: 1.455
Validation Accuracy: 55.90%
```

## 🔗 Testing Data Links

https://www.researchgate.net/figure/Examples-of-document-image-parsing-on-real-English-and-Chinese-documents-a-real_fig4_383460512
https://www.reddit.com/r/llc/comments/10ftdpi/is_this_a_scam_or_real_received_something_like/
https://graphicburger.com/a4-paper-psd-mockup/
https://elegantsi.com/blog/organizing-important-documents

## 💾 Dataset Source

https://www.kaggle.com/datasets/shaz13/real-world-documents-collections

## 📋 Output

Results are saved in `test_data/output/` with:</br>
Processed PDF documents
OCR text layer
output.json with metadata and classification results
