"""
image_folder.py
--------------------
A custom dataset class for loading images from a directory.
"""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    """
    A dataset that loads all images from a specified directory (recursively).
    Optionally applies a transform to each image.
    """

    def __init__(
        self,
        root_dir,
        transform=None,
        valid_extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
        disable_transform=False,
    ):
        """
        :param root_dir: The root directory containing images. Can be nested.
        :param transform: (Optional) A transform (e.g., torchvision transforms) to apply.
        :param valid_extensions: A tuple of valid image file extensions.
        :param disable_transform: If True, ignore the transform and return raw PIL images.
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
        self.image_paths = sorted(
            p for p in root_path.rglob("*") if p.suffix.lower() in self.valid_extensions
        )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No image files found in {root_dir} with extensions {self.valid_extensions}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of size {len(self)}."
            )

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if not self.disable_transform and self.transform is not None:
            # Apply dataset transform if allowed
            img = self.transform(img)

        return img, idx
