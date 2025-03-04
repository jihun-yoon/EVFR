"""
single_inference.py
------------------
Script for running inference on images using CPU or single GPU.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from evfr.embeddings.hf_dinov2 import HuggingFaceDinoV2Embedding
from evfr.datasets.image_folder import ImageFolderDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on images")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save embeddings",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help="Device to run inference on (default: auto-detect)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="HuggingFace model name (default: facebook/dinov2-base)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    print(f"Initializing model...")
    model = HuggingFaceDinoV2Embedding(
        model_name=args.model_name,
        device=args.device,
        use_processor=True
    )
    
    # Create dataset
    print(f"Loading images from {args.input_dir}")
    dataset = ImageFolderDataset(
        root_dir=args.input_dir,
        disable_transform=True  # Let the model handle transforms
    )
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = model.extract_dataset(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=4 if args.device == 'cuda' else 0
    )
    
    # Save results
    output_path = output_dir / "embeddings.npz"
    image_paths = [str(p) for p in dataset.image_paths]
    
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        image_paths=image_paths
    )
    
    print(f"Results saved to {output_path}")
    print(f"Embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()