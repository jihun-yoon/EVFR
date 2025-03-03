"""
download_tiny_iamgenet200.py
----------------
A script to download and extract the tiny-imagenet-200 dataset.

Usage:
    python download_tiny_imagenet200.py
"""

import requests
import zipfile
import os
from pathlib import Path

def download_tiny_imagenet(root_dir="../datasets/tiny-imagenet-200"):
    """Downloads and extracts the tiny-imagenet-200 dataset."""

    # Create the root directory if it doesn't exist
    os.makedirs(root_dir, exist_ok=True)

    # URLs for the dataset files (replace with actual URLs if needed)
    urls = [
        "http://cs231n.stanford.edu/tiny-imagenet-200.zip"  # Example URL
    ]

    for url in urls:
        filename = os.path.basename(url)
        filepath = os.path.join(root_dir, filename)

        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {filename}")

        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(root_dir)
        print(f"Extracted {filename}")
        # Clean up the zip file
        os.remove(filepath)

    print("Dataset download complete.")

if __name__ == "__main__":
    download_tiny_imagenet()

    # Create an ImageFolderDataset instance
    #dataset = ImageFolderDataset(root_dir="./tiny-imagenet-200/tiny-imagenet-200/train") #replace with correct path

    # Now you can use the 'dataset' object like any other PyTorch dataset
    # example access
    #print(f"Dataset size: {len(dataset)}")
    #image = dataset[0]  # Access the first image
    #print(image) # Print image info if necessary
