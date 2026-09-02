import torch
import torch.nn as nn


def dice(pred, target, smooth=1e-6):
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

    return dice_coeff.mean()


def IoU(pred, target, smooth=1e-6):
    """
    Compute the Intersection over Union (IoU) between the predicted and target tensors.

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
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

    # Compute the IoU coefficient
    iou_coeff = (intersection + smooth) / (union + smooth)

    # Compute the IoU loss
    iou_loss_value = 1 - iou_coeff.mean()

    return iou_loss_value


def precision(pred, target, smooth=1e-6):
    """
    Compute the Precision between the predicted and target tensors.

    Args:
        pred (torch.Tensor): The predicted tensor of shape (N, C, H, W).
        target (torch.Tensor): The target tensor of shape (N, C, H, W).
        smooth (float): A small value to avoid division by zero.
    """

    # Flatten the tensors
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # Compute the true positives and false positives
    true_positives = (pred_flat * target_flat).sum(dim=1)
    false_positives = (pred_flat * (1 - target_flat)).sum(dim=1)

    # Compute the Precision
    precision_value = (true_positives + smooth) / (true_positives + false_positives + smooth)

    return precision_value.mean()


def threshold_mask():
    pass


def connected_components():
    pass


def match_instances():
    pass


def average_precision():
    pass


def mean_average_precision():
    pass