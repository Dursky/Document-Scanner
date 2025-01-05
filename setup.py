from setuptools import setup, find_packages

setup(
    name="document_scanner",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'opencv-python>=4.5.0',
        'numpy>=1.19.0',
        'Pillow>=8.0.0',
        'img2pdf>=0.4.0',
    ],
    author="Michał Durski",
    description="A document scanning and processing tool",
    keywords="document, scanner, opencv, image processing",
    python_requires=">=3.7",
)