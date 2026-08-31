import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


from src.nn.models import UNet, train_loop, test_loop
from src.nn.optimizers import create_optimizer
from src.plot.plot import plot_resultados
from src.dataset.sintetic import SyntheticEllipses


# Definindo a seed e o device (GPU se disponível, caso contrário CPU)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Definindo o dataset sintético, dataloaders, modelo, função de perda e otimizador
dataset = SyntheticEllipses(n=120, size=128)
train_dataset, val_dataset = random_split(
    dataset,
    [100, 20],
    generator=torch.Generator().manual_seed(SEED)
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
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
NUM_EPOCHS = 5
for epoch in range(NUM_EPOCHS):
    plot_resultados(model, val_loader, device, num_images=4)
    train_loss = train_loop(train_loader, model, loss_fn, optimizer)
    val_loss, val_dice = test_loop(val_loader, model, loss_fn)
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Dice: {val_dice:.4f}"
    )