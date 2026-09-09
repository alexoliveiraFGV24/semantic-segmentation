import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(pred, target, class_weights=None):
    """
    Compute the Cross-Entropy loss between the predicted and target tensors.

    Args:
        pred (torch.Tensor): The predicted tensor of shape (N, C, H, W).
        target (torch.Tensor): The target tensor of shape (N, H, W) with class indices.
        class_weights (torch.Tensor): A tensor of shape (C,) containing the weights for each class.
    """

    if class_weights is not None:
        # Use PyTorch's built-in CrossEntropyLoss with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        # Use PyTorch's built-in CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
    loss_value = criterion(pred, target)

    return loss_value


def focal_loss(pred, target, class_weights=None, gamma=2.0):
    """
    Compute the Balanced Focal loss between the predicted and target tensors.

    Args:
        pred (torch.Tensor): The predicted tensor of shape (N, C, H, W).
        target (torch.Tensor): The target tensor of shape (N, H, W) with class indices.
        class_weights (torch.Tensor): A tensor of shape (C,) containing the weights for each class.
        alpha (float): The weighting factor for the class imbalance.
        gamma (float): The focusing parameter to reduce the loss for well-classified examples.
    """

    log_probs = F.log_softmax(pred, dim=1)

    log_pt = log_probs.gather(
        1,
        target.unsqueeze(1)
    ).squeeze(1)

    pt = log_pt.exp()

    loss = -(1 - pt) ** gamma * log_pt

    if class_weights is not None:
        loss = class_weights[target] * loss

    return loss.mean()


def dice_loss(pred, target, smooth=1e-6):
    """
    Compute the Dice coefficient between the predicted and target tensors.

    Args:
        pred (torch.Tensor): The predicted tensor of shape (N, C, H, W).
        target (torch.Tensor): The target tensor of shape (N, C, H, W).
        smooth (float): A small value to avoid division by zero.
    """

    # Flatten the tensors
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # Compute the intersection and union
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    # Compute the Dice coefficient
    dice_coeff = (2. * intersection + smooth) / (union + smooth)

    # Compute the Dice loss
    dice_loss_value = 1 - dice_coeff.mean()

    return dice_loss_value


def l1_loss(pred, target):
    """
    Compute the L1 loss (Mean Absolute Error) between the predicted and target tensors.

    Args:
        pred (torch.Tensor): The predicted tensor of shape (N, C, H, W).
        target (torch.Tensor): The target tensor of shape (N, C, H, W).
    """

    # Compute the L1 loss
    l1_loss_value = torch.nn.functional.l1_loss(pred, target)

    return l1_loss_value


def l2_loss(pred, target):
    """
    Compute the L2 loss (Mean Squared Error) between the predicted and target tensors.

    Args:
        pred (torch.Tensor): The predicted tensor of shape (N, C, H, W).
        target (torch.Tensor): The target tensor of shape (N, C, H, W).
    """

    # Compute the L2 loss
    l2_loss_value = torch.nn.functional.mse_loss(pred, target)

    return l2_loss_value


def discriminative_loss(
    embeddings,
    instance_maps,
    delta_v=0.5,
    delta_d=1.5,
    alpha=1.0,
    beta=1.0,
    gamma=0.001,
    max_pixels=8192,
):
    """
    Perda discriminativa da Trilha B (De Brabandere et al., 2017).

    Três termos, calculados por imagem e depois promediados no batch:

        variância  puxa cada pixel para o centróide do embedding da sua
                   própria instância, com margem delta_v (dentro da
                   margem o gradiente é zero);

        distância  empurra centróides de instâncias diferentes para
                   longe, com margem 2 * delta_d;

        regularização  mantém os centróides perto da origem, para os
                   embeddings não escaparem para o infinito.

    Só os pixels de foreground entram: o fundo não tem identidade de
    instância, e incluí-lo criaria um "cluster de fundo" gigante que
    domina o termo de variância.

    Args:
        embeddings: (N, D, H, W) embedding por pixel.
        instance_maps: (N, H, W) inteiros, 0 = fundo, 1.. = instâncias.
        delta_v: margem de atração. Raio máximo desejado de um cluster.
        delta_d: margem de repulsão. Use delta_d > 2 * delta_v para que
            os clusters fiquem separáveis por uma bola de raio delta_v.
        alpha, beta, gamma: pesos dos termos de variância, distância e
            regularização.
        max_pixels: teto de pixels de foreground amostrados por imagem.
            Limita a memória do grafo de autograd sem enviesar a perda.

    Returns:
        (loss, parts) com parts = {"var": float, "dist": float, "reg": float}.
    """

    batch_size, embedding_dim = embeddings.shape[0], embeddings.shape[1]

    variance_total = embeddings.new_zeros(())
    distance_total = embeddings.new_zeros(())
    regularization_total = embeddings.new_zeros(())

    n_valid = 0

    for i in range(batch_size):

        flat_embedding = embeddings[i].reshape(embedding_dim, -1)
        flat_instances = instance_maps[i].reshape(-1)

        foreground = torch.nonzero(flat_instances > 0, as_tuple=False).squeeze(1)

        if foreground.numel() == 0:
            continue

        # Subamostra para limitar o tamanho do grafo em imagens densas.
        if foreground.numel() > max_pixels:
            choice = torch.randperm(
                foreground.numel(),
                device=foreground.device
            )[:max_pixels]

            foreground = foreground[choice]

        pixels = flat_embedding[:, foreground].t()      # (P, D)
        ids = flat_instances[foreground]                # (P,)

        unique_ids = torch.unique(ids)
        n_instances = unique_ids.numel()

        # (K, P): qual pixel pertence a qual instância
        membership = (
            ids.unsqueeze(0) == unique_ids.unsqueeze(1)
        ).to(pixels.dtype)

        counts = membership.sum(dim=1).clamp(min=1.0)                # (K,)
        centroids = (membership @ pixels) / counts.unsqueeze(1)      # (K, D)

        # -------------------------------------------------------
        # VARIÂNCIA  (atração)
        # -------------------------------------------------------

        distances = torch.cdist(centroids, pixels)                   # (K, P)

        pull = torch.clamp(distances - delta_v, min=0.0) ** 2

        variance = ((pull * membership).sum(dim=1) / counts).mean()

        # -------------------------------------------------------
        # DISTÂNCIA  (repulsão)
        # -------------------------------------------------------

        if n_instances > 1:

            centroid_distances = torch.cdist(centroids, centroids)   # (K, K)

            push = torch.clamp(
                2 * delta_d - centroid_distances,
                min=0.0
            ) ** 2

            off_diagonal = ~torch.eye(
                n_instances,
                dtype=torch.bool,
                device=embeddings.device
            )

            distance = (
                push[off_diagonal].sum()
                /
                (n_instances * (n_instances - 1))
            )

        else:
            # Uma única instância não tem par para empurrar.
            distance = embeddings.new_zeros(())

        # -------------------------------------------------------
        # REGULARIZAÇÃO
        # -------------------------------------------------------

        regularization = centroids.norm(dim=1).mean()

        variance_total = variance_total + variance
        distance_total = distance_total + distance
        regularization_total = regularization_total + regularization

        n_valid += 1

    if n_valid == 0:
        # Nenhuma instância no batch: perda zero, mas conectada ao grafo.
        zero = embeddings.sum() * 0.0
        return zero, {"var": 0.0, "dist": 0.0, "reg": 0.0}

    variance_total = variance_total / n_valid
    distance_total = distance_total / n_valid
    regularization_total = regularization_total / n_valid

    loss = (
        alpha * variance_total
        +
        beta * distance_total
        +
        gamma * regularization_total
    )

    parts = {
        "var": float(variance_total.detach()),
        "dist": float(distance_total.detach()),
        "reg": float(regularization_total.detach()),
    }

    return loss, parts