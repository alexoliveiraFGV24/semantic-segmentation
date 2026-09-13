import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()

        # padding=dilation mantem a resolucao espacial (kernel 3x3)
        # tanto no caso normal (dilation=1, padding=1) quanto no
        # atrous (dilation>1, padding=dilation).
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# SEGNET
# ============================================================

class SegNet(nn.Module):
    """
    Mecanismo de recuperacao de resolucao: pool indices (max unpooling,
    slides 14 e 16).

    Ao contrario da U-Net, o decoder NAO recebe nenhuma feature do
    encoder por concatenacao. A unica informacao espacial que atravessa
    o gargalo sao os INDICES de onde cada maximo do max-pooling veio;
    o MaxUnpool2d devolve cada valor ao seu lugar exato e preenche o
    resto com zero -- sem aprender nada extra na subida.

    Larguras identicas as da U-Net (32/64/128) para que a Parte 3,
    Eixo 1, compare apenas o mecanismo de resolucao, nao a capacidade.
    """

    def __init__(self, num_classes):
        super().__init__()

        self.enc1 = ConvBlock(3, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

        # Decoder: unpool -> conv. Sem concatenacao com o encoder.
        self.dec3 = ConvBlock(128, 64)
        self.dec2 = ConvBlock(64, 32)
        self.dec1 = ConvBlock(32, 32)

        self.final_conv = nn.Conv2d(32, num_classes, 1)

    def forward_features(self, x):

        e1 = self.enc1(x)
        p1, idx1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2, idx2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3, idx3 = self.pool(e3)

        d3 = self.unpool(p3, idx3, output_size=e3.shape[-2:])
        d3 = self.dec3(d3)

        d2 = self.unpool(d3, idx2, output_size=e2.shape[-2:])
        d2 = self.dec2(d2)

        d1 = self.unpool(d2, idx1, output_size=e1.shape[-2:])
        d1 = self.dec1(d1)

        return d1

    def forward(self, x):
        return self.final_conv(self.forward_features(x))


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.enc1 = ConvBlock(3, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2, 2)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec3 = ConvBlock(192, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec2 = ConvBlock(96, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.dec1 = ConvBlock(48, 16)

        self.final_conv = nn.Conv2d(16, num_classes, 1)

    def forward_features(self, x):

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        # Decoder + skip connections
        d3 = self.up3(p3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        # Não há skip adicional aqui
        # pois a resolução já voltou ao tamanho original
        d1 = self.dec1(
            torch.cat([d1, e1], dim=1)
        )

        return d1

    def forward(self, x):
        return self.final_conv(self.forward_features(x))



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
    """
    Mesmo mecanismo de recuperacao de resolucao que a U-Net (skip
    connections, slides 25-29): mesmas larguras (32/64/128) e a MESMA
    arvore de concatenacoes (192->64, 96->32, 48->16). A unica
    diferenca e trocar ConvBlock por ResidualBlock -- isola "bloco
    residual" como a variavel, com o mecanismo de resolucao fixo.
    """

    def __init__(self, num_classes):
        super().__init__()

        self.enc1 = ResidualBlock(3, 32)
        self.enc2 = ResidualBlock(32, 64)
        self.enc3 = ResidualBlock(64, 128)

        self.pool = nn.MaxPool2d(2, 2)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec3 = ResidualBlock(192, 64)      # 64 (up) + 128 (skip e3)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec2 = ResidualBlock(96, 32)       # 32 (up) + 64 (skip e2)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.dec1 = ResidualBlock(48, 16)       # 16 (up) + 32 (skip e1)

        self.final_conv = nn.Conv2d(16, num_classes, 1)

    def forward_features(self, x):

        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        d3 = self.up3(p3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return d1

    def forward(self, x):
        return self.final_conv(self.forward_features(x))


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
    """
    Mecanismo de recuperacao de resolucao: atrous convolution
    mantendo o output stride (slides 39-42).

    UNet/SegNet/ResUNet reduzem a resolucao 3 vezes (output stride 8)
    e precisam de 3 etapas de upsampling aprendido para voltar ao
    tamanho original. Aqui so ha UM max-pool (output stride 2); a
    3a camada do encoder, no lugar de um 2o pool, usa convolucao
    DILATADA (dilation=2) na mesma resolucao -- o campo receptivo
    cresce sem perder resolucao espacial. So resta UM upsample
    bilinear (sem parametros) para fechar a conta.
    """

    def __init__(self, num_classes):
        super().__init__()

        self.enc1 = ConvBlock(3, 32)              # resolucao total
        self.pool = nn.MaxPool2d(2, 2)             # UNICA reducao (output stride 2)
        self.enc2 = ConvBlock(32, 64)               # 1/2 da resolucao
        self.enc3 = ConvBlock(64, 128, dilation=2)   # atrous no lugar do 2o pool

        self.aspp = ASPP(128, 64)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward_features(self, x):

        original_size = x.shape[-2:]

        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        e3 = self.enc3(e2)          # dilatada: resolucao preservada

        feat = self.aspp(e3)

        # Recupera resolucao original (unico upsample do modelo).
        # Como e um conv 1x1 quem vem depois, fazer o upsample nas
        # features ou nos logits da exatamente o mesmo resultado.
        feat = F.interpolate(
            feat,
            size=original_size,
            mode="bilinear",
            align_corners=False
        )

        return feat

    def forward(self, x):
        return self.classifier(self.forward_features(x))


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
    """
    Contexto global (Eixo 3, slides 51-55): pyramid pooling em varias
    escalas, concatenado de volta as features locais antes de
    classificar. Nao entra na Parte 3 desta entrega (que usa os
    Eixos 1 e 2), mas as larguras seguem o mesmo esquema (32/64/128)
    dos demais backbones para ficar pronto quando o Eixo 3 for feito.
    """

    def __init__(self, num_classes):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.psp = PyramidPooling(
            in_channels=128,
            out_channels=32
        )

        # 128 + 4*32 = 256
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(128, num_classes, 1)

    def forward_features(self, x):

        original_size = x.shape[-2:]

        x = self.backbone(x)
        x = self.psp(x)
        x = self.head(x)

        x = F.interpolate(
            x,
            size=original_size,
            mode="bilinear",
            align_corners=False
        )

        return x

    def forward(self, x):
        return self.classifier(self.forward_features(x))


# ============================================================
# TRILHA A  --  duas cabecas sobre qualquer backbone
# ============================================================

BACKBONES = {
    "segnet":  SegNet,
    "unet":    UNet,
    "resunet": ResUNet,
    "deeplab": DeepLabV3,
    "pspnet":  PSPNet,
}

# canais de `forward_features` de cada backbone
FEATURE_CHANNELS = {
    "segnet": 32, "unet": 16, "resunet": 16, "deeplab": 64, "pspnet": 128,
}


class TrilhaAModel(nn.Module):
    """
    Modelo da Trilha A (Parte 2): duas cabecas 1x1 sobre as features
    de qualquer backbone acima.

        semantic : [N, 3, H, W]  logits  (0=fundo, 1=interior, 2=fronteira)
        distance : [N, 1, H, W]  logit   (sigmoid -> distancia normalizada 0..1)

    E a mesma troca "so o final_conv muda" do 2_instances.ipynb; mora
    aqui (e nao so no notebook) para que o checkpoint salvo na Parte 4
    seja carregavel pelo inferencia.ipynb e pelas Partes 5-6 sem copiar
    a classe.
    """

    def __init__(self, backbone_kind="unet"):
        super().__init__()

        self.backbone_kind = backbone_kind
        self.backbone = BACKBONES[backbone_kind](num_classes=2)  # num_classes p/ compat; nao usado

        c = FEATURE_CHANNELS[backbone_kind]
        self.semantic_head = nn.Conv2d(c, 3, kernel_size=1)
        self.distance_head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        feat = self.backbone.forward_features(x)
        return self.semantic_head(feat), self.distance_head(feat)


# ============================================================
# Loops de treino e teste
# ============================================================


def train_loop(dataloader, device, modelo, loss_fc, otimizador):

    modelo.train()
    total_loss = 0

    for X, y in dataloader:

        X = X.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )

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

            X = X.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )
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