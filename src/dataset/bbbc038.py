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
    """
    Dataset para o BBBC038 / Data Science Bowl 2018.

    Estrutura esperada:

    root/
    ├── id_1/
    │   ├── images/
    │   │   └── image.png
    │   └── masks/
    │       ├── mask_1.png
    │       ├── mask_2.png
    │       └── ...
    │
    ├── id_2/
    │   ├── images/
    │   │   └── image.png
    │   └── masks/
    │       └── ...

    Retorno:
        image -> Tensor [C, H, W]
        mask  -> Tensor [H, W]
    """

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None
    ):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform

        self.samples = self._find_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Nenhuma amostra encontrada em: {self.root}"
            )

    def _find_samples(self):
        samples = []

        # Cada subpasta representa uma imagem
        for image_dir in sorted(self.root.iterdir()):

            if not image_dir.is_dir():
                continue

            images_dir = image_dir / "images"
            masks_dir = image_dir / "masks"

            if not images_dir.exists():
                continue

            if not masks_dir.exists():
                continue

            image_files = list(images_dir.glob("*"))

            # Normalmente existe uma única imagem
            image_files = [
                f for f in image_files
                if f.suffix.lower() in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".tif",
                    ".tiff"
                }
            ]

            if len(image_files) == 0:
                continue

            image_path = image_files[0]

            mask_files = [
                f for f in masks_dir.iterdir()
                if f.suffix.lower() == ".png"
            ]

            samples.append({
                "image": image_path,
                "masks": sorted(mask_files)
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        """
        Carrega a imagem como RGB.
        """

        image = Image.open(path).convert("RGB")

        return image

    def _load_mask(self, mask_paths, size):
        """
        Junta todas as máscaras individuais
        em uma única máscara binária.

        0 = background
        1 = nucleus
        """

        width, height = size

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        for mask_path in mask_paths:

            nucleus = Image.open(mask_path).convert("L")

            nucleus = np.array(nucleus)

            # Qualquer pixel diferente de zero
            # pertence ao núcleo
            mask[nucleus > 0] = 1

        return Image.fromarray(mask)

    def __getitem__(self, index):

        sample = self.samples[index]

        image = self._load_image(
            sample["image"]
        )

        mask = self._load_mask(
            sample["masks"],
            image.size
        )

        # Transformações
        if self.transform is not None:

            image, mask = self.transform(
                image,
                mask
            )

        else:

            image = torch.from_numpy(
                np.array(image)
            ).permute(2, 0, 1).float() / 255.0

            mask = torch.from_numpy(
                np.array(mask)
            ).long()

        if self.target_transform is not None:
            mask = self.target_transform(mask)

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