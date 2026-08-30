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