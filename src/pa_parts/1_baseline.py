import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


from src.dataset.bbbc038 import BBBC038Dataset, SegmentationTransform
from src.nn.models import UNet, train_loop, test_loop
from src.nn.optimizers import create_optimizer
from src.plot.plot import plot_resultados
from src.nn.metrics import dice, IoU, precision, average_precision, mean_average_precision



transform = SegmentationTransform(
    size=(256, 256)
)

dataset = BBBC038Dataset(
    root="data/stage1_train",
    transform=transform
)


# ============== PARAMETROS ====================
train_len = int(0.8 * len(dataset))
val_len = int(0.2 * len(dataset))
batch_size = 32
SEED = 67
NUM_EPOCHS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8
)


# Definindo a seed e o device (GPU se disponível, caso contrário CPU)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Definindo o dataset sintético, dataloaders, modelo, função de perda e otimizador
train_dataset, val_dataset = random_split(
    dataset,
    [train_len, val_len],
    generator=torch.Generator().manual_seed(SEED)
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
model = UNet(num_classes=2).to(device)

# Funções de perda ponderadas para lidar com o desbalanceamento das classes
# Classe 0 = background
# Classe 1 = ellipse
# Damos menos peso ao background e mais peso às elipses.
weights = torch.tensor([0.25, 1.75],dtype=torch.float32).to(device)
loss_fn = nn.CrossEntropyLoss(weight=weights)

# Otimizador
optimizer = create_optimizer("Adam", model, lr=1e-3)

# Treinamento e validação do modelo
for epoch in range(NUM_EPOCHS):
    train_loss = train_loop(train_loader, device, model, loss_fn, optimizer)
    val_loss, pixel_acc = test_loop(val_loader, device, model, loss_fn)
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Pixel Accuracy: {pixel_acc:.4f}"
    )
    plot_resultados(model, val_loader, device, num_images=4)

