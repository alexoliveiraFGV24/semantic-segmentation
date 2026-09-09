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
        transform=None,
        mode="semantic"
    ):
        """
        Args:
            root:
                Caminho para stage1_train.

            transform:
                Transformação aplicada simultaneamente à imagem
                e ao target.

            mode:
                "semantic"  -> máscara binária
                "instance"  -> mapa de instâncias
        """

        if mode not in {"semantic", "instance"}:
            raise ValueError(
                "mode deve ser 'semantic' ou 'instance'"
            )

        self.root = Path(root)
        self.transform = transform
        self.mode = mode

        self.samples = self._find_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Nenhuma amostra encontrada em: {self.root}"
            )

    # ==========================================================
    # ENCONTRA AS AMOSTRAS
    # ==========================================================

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
                f
                for f in images_dir.iterdir()
                if f.suffix.lower() in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".tif",
                    ".tiff"
                }
            ]

            mask_files = [
                f
                for f in masks_dir.iterdir()
                if f.suffix.lower() == ".png"
            ]

            if len(image_files) == 0:
                continue

            samples.append({
                "image": image_files[0],
                "masks": sorted(mask_files)
            })

        return samples

    # ==========================================================
    # TAMANHO
    # ==========================================================

    def __len__(self):
        return len(self.samples)

    # ==========================================================
    # CARREGA IMAGEM
    # ==========================================================

    def _load_image(self, path):

        return Image.open(path).convert("RGB")

    # ==========================================================
    # MÁSCARA SEMÂNTICA
    # ==========================================================

    def _load_semantic_mask(
        self,
        mask_paths,
        image_size
    ):
        """
        Retorna:

            0 = background
            1 = foreground

        Formato:
            [H, W]
        """

        width, height = image_size

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        for mask_path in mask_paths:

            nucleus = Image.open(
                mask_path
            ).convert("L")

            nucleus = np.array(
                nucleus
            )

            mask[
                nucleus > 0
            ] = 1

        return Image.fromarray(mask)

    # ==========================================================
    # MAPA DE INSTÂNCIAS
    # ==========================================================

    def _load_instance_map(
        self,
        mask_paths,
        image_size
    ):
        """
        Retorna:

            0 = background
            1 = instância 1
            2 = instância 2
            3 = instância 3
            ...

        Formato:
            [H, W]
        """

        width, height = image_size

        instance_map = np.zeros(
            (height, width),
            dtype=np.int32
        )

        for instance_id, mask_path in enumerate(
            mask_paths,
            start=1
        ):

            nucleus = Image.open(
                mask_path
            ).convert("L")

            nucleus = np.array(
                nucleus
            )

            instance_map[
                nucleus > 0
            ] = instance_id

        return Image.fromarray(
            instance_map
        )

    # ==========================================================
    # MÁSCARAS INDIVIDUAIS
    # ==========================================================

    def _load_instance_masks(
        self,
        mask_paths
    ):
        """
        Retorna uma lista de máscaras booleanas,
        uma para cada instância.

        Útil para avaliação.
        """

        instances = []

        for mask_path in mask_paths:

            nucleus = Image.open(
                mask_path
            ).convert("L")

            nucleus = np.array(
                nucleus
            )

            instances.append(
                nucleus > 0
            )

        return instances

    # ==========================================================
    # GETITEM
    # ==========================================================

    def __getitem__(self, index):

        sample = self.samples[index]

        image = self._load_image(
            sample["image"]
        )

        # ------------------------------------------------------
        # Escolhe o target
        # ------------------------------------------------------

        if self.mode == "semantic":

            target = self._load_semantic_mask(
                sample["masks"],
                image.size
            )

        else:

            target = self._load_instance_map(
                sample["masks"],
                image.size
            )

        # ------------------------------------------------------
        # Transform
        # ------------------------------------------------------

        if self.transform is not None:

            image, target = self.transform(
                image,
                target
            )

        else:

            image = TF.to_tensor(image)

            target = torch.from_numpy(
                np.array(target)
            ).long()

        return image, target


# ==========================================================
# TRANSFORM
# ==========================================================

class SegmentationTransform:

    def __init__(
        self,
        size=(256, 256),
        augment=False
    ):

        self.size = size
        self.augment = augment

    def __call__(
        self,
        image,
        target
    ):

        # ------------------------------------------------------
        # Resize
        # ------------------------------------------------------

        image = TF.resize(
            image,
            self.size,
            interpolation=TF.InterpolationMode.BILINEAR
        )

        target = TF.resize(
            target,
            self.size,
            interpolation=TF.InterpolationMode.NEAREST
        )

        # ------------------------------------------------------
        # Augmentation
        # ------------------------------------------------------

        if self.augment:

            if random.random() > 0.5:

                image = TF.hflip(image)
                target = TF.hflip(target)

            if random.random() > 0.5:

                image = TF.vflip(image)
                target = TF.vflip(target)

        # ------------------------------------------------------
        # Image
        # ------------------------------------------------------

        image = TF.to_tensor(image)

        # ------------------------------------------------------
        # Target
        # ------------------------------------------------------

        target = np.array(target)

        # ------------------------------------------------------
        # Para semantic:
        #
        # 0, 1, 2, 3, ... não pode existir.
        #
        # Garantimos que seja 0/1.
        #
        # Isso não deve ser aplicado ao instance map!
        # ------------------------------------------------------

        if np.max(target) > 1:
            # Só fazemos isso se necessário?
            #
            # NÃO podemos fazer isso para instance mode.
            pass

        target = torch.from_numpy(
            target
        ).long()

        return image, target