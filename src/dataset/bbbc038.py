"""
from torch.utils.data import DataLoader


transform = SegmentationTransform(
    size=(256, 256)
)

dataset = BBBC038Dataset(
    root="data/stage1_train",
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)

"""


from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class BBBC038Dataset(Dataset):

    def __init__(
        self,
        root,
        transform=None
    ):
        self.root = Path(root)
        self.transform = transform

        self.samples = self._find_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Nenhuma amostra encontrada em: {self.root}"
            )

    def _find_samples(self):

        samples = []

        for image_dir in sorted(self.root.iterdir()):

            if not image_dir.is_dir():
                continue

            images_dir = image_dir / "images"
            masks_dir = image_dir / "masks"

            if not images_dir.exists():
                continue

            if not masks_dir.exists():
                continue

            image_files = [
                f for f in images_dir.iterdir()
                if f.suffix.lower() in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".tif",
                    ".tiff"
                }
            ]

            mask_files = [
                f for f in masks_dir.iterdir()
                if f.suffix.lower() == ".png"
            ]

            if not image_files:
                continue

            samples.append({
                "image": image_files[0],
                "masks": sorted(mask_files)
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        return Image.open(path).convert("RGB")

    def _load_mask(self, mask_paths, image_size):

        width, height = image_size

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        for mask_path in mask_paths:

            nucleus = Image.open(
                mask_path
            ).convert("L")

            nucleus = np.array(nucleus)

            mask[nucleus > 0] = 1

        return Image.fromarray(mask)

    def _load_instance_masks(self, mask_paths, size):

        width, height = size

        instances = []

        for mask_path in mask_paths:

            nucleus = Image.open(
                mask_path
            ).convert("L")

            nucleus = TF.resize(
                nucleus,
                (height, width),
                interpolation=TF.InterpolationMode.NEAREST
            )

            nucleus = np.array(nucleus)

            instances.append(
                nucleus > 0
            )

        return instances

    def __getitem__(self, index):

        sample = self.samples[index]

        image = self._load_image(
            sample["image"]
        )

        mask = self._load_mask(
            sample["masks"],
            image.size
        )

        if self.transform is not None:

            image, mask = self.transform(
                image,
                mask
            )

        else:

            # Imagem: [C, H, W]
            image = (
                torch.from_numpy(
                    np.array(image)
                )
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

            # Máscara: [H, W]
            mask = torch.from_numpy(
                np.array(mask)
            ).long()

        return image, mask


class SegmentationTransform:

    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, image, mask):

        # Resize
        image = TF.resize(
            image,
            self.size,
            interpolation=TF.InterpolationMode.BILINEAR
        )

        mask = TF.resize(
            mask,
            self.size,
            interpolation=TF.InterpolationMode.NEAREST
        )

        # Flip horizontal
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Flip vertical
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Tensor da imagem
        image = TF.to_tensor(image)

        # Tensor da máscara
        mask = torch.from_numpy(
            np.array(mask)
        ).long()

        return image, mask