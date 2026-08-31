import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# SEGNET
# ============================================================

class SegNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = ConvBlock(128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = ConvBlock(64, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec1 = ConvBlock(64, 64)

        self.final_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):

        # Encoder
        x = self.enc1(x)
        x = self.pool(x)

        x = self.enc2(x)
        x = self.pool(x)

        x = self.enc3(x)
        x = self.pool(x)

        # Decoder
        x = self.up3(x)
        x = self.dec3(x)

        x = self.up2(x)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        return self.final_conv(x)


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(2, 2)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec1 = ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        # Decoder + skip connections
        d3 = self.up3(p3)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        # Não há skip adicional aqui
        # pois a resolução já voltou ao tamanho original
        d1 = self.dec1(
            torch.cat([d1, d1], dim=1)
        )

        return self.final_conv(d1)


# ============================================================
# RESUNET
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, 1
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Encoder
        self.enc1 = ResidualBlock(3, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)

        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = ResidualBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = ResidualBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec1 = ResidualBlock(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):

        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        d3 = self.up3(p3)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        return self.final_conv(d1)


# ============================================================
# DEEPLABV3
# ============================================================

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=6,
                dilation=6
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=12,
                dilation=12
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=18,
                dilation=18
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                out_channels * 4,
                out_channels,
                1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        return self.project(x)


class DeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Backbone simplificado
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(256, 128)

        self.classifier = nn.Conv2d(
            128,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        original_size = x.shape[-2:]

        x = self.backbone(x)
        x = self.aspp(x)
        x = self.classifier(x)

        # Recupera resolução original
        x = F.interpolate(
            x,
            size=original_size,
            mode="bilinear",
            align_corners=False
        )

        return x


# ============================================================
# PSPNET
# ============================================================

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.pool_sizes = [1, 2, 3, 6]

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for size in self.pool_sizes
        ])

    def forward(self, x):

        h, w = x.shape[-2:]

        outputs = [x]

        for block in self.blocks:

            pooled = block(x)

            pooled = F.interpolate(
                pooled,
                size=(h, w),
                mode="bilinear",
                align_corners=False
            )

            outputs.append(pooled)

        return torch.cat(outputs, dim=1)


class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.psp = PyramidPooling(
            in_channels=256,
            out_channels=64
        )

        # 256 + 4*64 = 512
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                num_classes,
                1
            )
        )

    def forward(self, x):

        original_size = x.shape[-2:]

        x = self.backbone(x)

        x = self.psp(x)

        x = self.classifier(x)

        x = F.interpolate(
            x,
            size=original_size,
            mode="bilinear",
            align_corners=False
        )

        return x


# ============================================================
# Loops de treino e teste
# ============================================================


def train_loop(dataloader, device, modelo, loss_fc, otimizador):

    modelo.train()
    total_loss = 0

    for X, y in dataloader:

        X, y = X.to(device), y.to(device)

        pred = modelo(X)
        loss = loss_fc(pred, y)

        otimizador.zero_grad()
        loss.backward()
        otimizador.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)



def test_loop(dataloader, device, modelo, loss_fc):

    modelo.eval()

    total_loss = 0
    corretas = 0
    total_pixels = 0

    with torch.no_grad():

        for X, y in dataloader:

            X, y = X.to(device), y.to(device)

            outputs = modelo(X)

            loss = loss_fc(outputs, y)
            total_loss += loss.item()

            pred = outputs.argmax(dim=1)

            corretas += (pred == y).sum().item()
            total_pixels += y.numel()

    return (
        total_loss / len(dataloader),
        corretas / total_pixels
    )