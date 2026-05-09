"""Setup script for the VexDR package."""

from pathlib import Path

from setuptools import find_namespace_packages, setup

ROOT = Path(__file__).parent


setup(
    name="VexDR",
    version="0.0.1",
    description="",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Phuong Ho",
    author_email="nho@tudelft.nl",
    python_requires=">=3.12",
    packages=find_namespace_packages(include=["VexDR*"]),
    include_package_data=True,
    install_requires=[
        "lightning==2.6.1",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "numpy",
        "pandas==3.0.1",
        "PyYAML>=6",
        "opencv-python-headless==4.13.0.92",
        "openslide-python==1.4.3",
        "openslide-bin==4.0.0.13",
        "timm==1.0.25",
        "python-dotenv==1.2.2",
        "matplotlib==3.10.7",
        "scikit-learn==1.8.0",
        "umap-learn==0.5.12",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
    ],
)
