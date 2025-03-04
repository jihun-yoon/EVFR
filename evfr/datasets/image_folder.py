"""
image_folder.py
--------------
Dataset implementation for loading and processing images from a directory structure.
"""

from pathlib import Path
from typing import Tuple, Optional, Callable, List
from PIL import Image
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    """Dataset for loading images from a directory with optional transformations.
    
    Recursively loads images from a root directory, supporting various image formats
    and optional preprocessing transforms.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
        disable_transform: bool = False,
    ) -> None:
        """Initialize the image folder dataset.

        Args:
            root_dir: Root directory containing image files
            transform: Optional transform function to preprocess images
            valid_extensions: Supported image file extensions
            disable_transform: Flag to bypass transform pipeline

        Raises:
            ValueError: If root_dir is not a valid directory
            FileNotFoundError: If no valid images found in directory
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.valid_extensions = valid_extensions
        self.disable_transform = disable_transform

        root_path = Path(root_dir)
        if not root_path.is_dir():
            raise ValueError(f"Provided path {root_dir} is not a valid directory.")

        # Recursively gather image files
        self.image_paths: List[Path] = sorted(
            p for p in root_path.rglob("*") if p.suffix.lower() in self.valid_extensions
        )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No image files found in {root_dir} with extensions {self.valid_extensions}"
            )

    def __len__(self) -> int:
        """Return the total number of images in dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image | any, int]:
        """Get an image and its index from the dataset.

        Args:
            idx: Index of the image to retrieve

        Returns:
            Tuple of (transformed image, index)
            Image type depends on transform implementation

        Raises:
            IndexError: If idx is out of valid range
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of size {len(self)}."
            )

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if not self.disable_transform and self.transform is not None:
            img = self.transform(img)

        return img, idx
