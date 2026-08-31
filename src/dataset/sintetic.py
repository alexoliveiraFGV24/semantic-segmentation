import random
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class SyntheticEllipses(Dataset):

    def __init__(self, n=120, size=128):

        self.images = []
        self.masks = []

        for _ in range(n):

            background = random.randint(20, 80)
            image = Image.new("L", (size, size), background)

            # Máscara binária
            mask = Image.new("L", (size, size), 0)

            draw_image = ImageDraw.Draw(image)
            draw_mask = ImageDraw.Draw(mask)

            num_ellipses = random.randint(5, 20)

            # Guardamos centros já utilizados para aumentar
            # a chance de criar elipses próximas/tocando
            centers = []

            for _ in range(num_ellipses):

                if centers and random.random() < 0.65:

                    # Escolhe uma elipse anterior e coloca
                    # a nova relativamente próxima dela
                    old_x, old_y = random.choice(centers)
                    cx = old_x + random.randint(-15, 15)
                    cy = old_y + random.randint(-15, 15)

                else:

                    cx = random.randint(10, size - 11)
                    cy = random.randint(10, size - 11)

                # Tamanho da elipse
                rx = random.randint(4, 16)
                ry = random.randint(4, 16)

                box = (
                    cx - rx,
                    cy - ry,
                    cx + rx,
                    cy + ry
                )

                # Intensidade da elipse
                intensity = random.randint(150, 255)

                # Desenha a elipse na imagem
                draw_image.ellipse(box, fill=intensity)

                # Desenha a elipse na máscara
                draw_mask.ellipse(box, fill=255)

                centers.append((cx, cy))

            # Contraste variável
            image_array = np.asarray(image, dtype=np.float32) / 255.0
            contrast = random.uniform(0.7, 1.4)
            brightness = random.uniform(-0.1, 0.1)
            image_array = image_array * contrast + brightness
            image_array = np.clip(image_array, 0, 1)

            # Ruído
            noise_std = random.uniform(0.0, 0.10)
            noise = np.random.normal(0, noise_std, image_array.shape)
            image_array += noise
            image_array = np.clip(image_array, 0, 1)

            # Converte grayscale -> RGB
            image_array = np.stack(
                [
                    image_array,
                    image_array,
                    image_array
                ],
                axis=0
            )
            image_tensor = torch.from_numpy(image_array).float()

            # Máscara
            mask_array = np.asarray(mask)
            mask_array = (mask_array > 0).astype(np.int64)
            mask_tensor = torch.from_numpy(mask_array)

            self.images.append(image_tensor)
            self.masks.append(mask_tensor)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        return (
            self.images[index],
            self.masks[index]
        )
