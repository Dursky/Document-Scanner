from setuptools import setup, find_namespace_packages

setup(
    name="document_scanner",
    version="0.1.0",
    package_dir={"document_scanner": "src"},
    packages=find_namespace_packages(include=["src.*"]),
   install_requires=[
        'opencv-python>=4.5.0,<5.0.0',
        'numpy>=1.19.0,<2.0.0',
        'Pillow>=8.0.0,<10.0.0',
        'reportlab>=3.6.8,<4.0.0', 
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0,<7.0.0',
            'pytest-cov>=2.12.0,<3.0.0',
            'flake8>=3.9.0,<4.0.0',
            'pylint>=2.8.0,<3.0.0',
        ],
    },
    author="Michał Durski",
    description="A document scanning and processing tool",
    keywords="document, scanner, opencv, image processing",
    python_requires=">=3.7",
)