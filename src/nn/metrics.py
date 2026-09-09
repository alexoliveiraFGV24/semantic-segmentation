import torch
import torch.nn as nn
from scipy import ndimage
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image


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

    Devolve o COEFICIENTE (quanto maior, melhor), não a perda. Para usar
    como perda, chame iou_loss().

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

    return iou_coeff.mean()


def iou_loss(pred, target, smooth=1e-6):
    """
    1 - IoU. Este é o valor que se minimiza no treino.
    """

    return 1 - IoU(pred, target, smooth=smooth)


def instance_IoU(pred, target):
    """
    IoU entre duas máscaras de instância.
    """

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    if union == 0:
        return 0.0

    return intersection / union


def binary_iou_dice(pred_mask, target_mask):
    """
    IoU e Dice semânticos entre duas máscaras binárias de numpy.

    É a métrica da Parte 1 (fundo vs. objeto), calculada sobre arrays
    booleanos já limiarizados — sem os cuidados de shape das versões
    em tensor acima.

    Args:
        pred_mask: array booleano [H, W].
        target_mask: array booleano [H, W].

    Returns:
        (iou, dice) como floats.
    """

    intersection = np.logical_and(pred_mask, target_mask).sum()

    pred_area = pred_mask.sum()
    target_area = target_mask.sum()

    union = pred_area + target_area - intersection

    if union == 0:
        # Ambas vazias: acerto perfeito por convenção.
        return 1.0, 1.0

    iou = intersection / union
    dice = (2 * intersection) / (pred_area + target_area + 1e-6)

    return float(iou), float(dice)


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


def connected_components(mask):
    """
    Extrai instâncias de uma máscara binária através
    de componentes conexos.

    Args:
        mask: numpy array [H, W], booleano ou 0/1

    Returns:
        Lista de máscaras booleanas, uma por instância.
    """

    labeled, num_instances = ndimage.label(mask)

    instances = []

    for instance_id in range(1, num_instances + 1):
        instances.append(
            labeled == instance_id
        )

    return instances


def match_instances(pred_instances, target_instances, iou_threshold):
    """
    Matching guloso por IoU decrescente.

    Cada previsão pode ser associada a no máximo
    uma instância verdadeira, e cada ground truth
    a no máximo uma previsão.

    Returns:
        tp, fp, fn
    """

    if len(pred_instances) == 0:
        return (0, 0, len(target_instances))

    if len(target_instances) == 0:
        return (0, len(pred_instances), 0)

    pairs = []

    for pred_idx, pred in enumerate(pred_instances):

        for target_idx, target in enumerate(target_instances):

            iou = instance_IoU(pred, target)

            if iou >= iou_threshold:
                pairs.append((iou, pred_idx, target_idx))

    # Maior IoU primeiro
    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_pred = set()
    matched_target = set()

    tp = 0

    for iou, pred_idx, target_idx in pairs:

        if pred_idx in matched_pred:
            continue

        if target_idx in matched_target:
            continue

        matched_pred.add(pred_idx)
        matched_target.add(target_idx)

        tp += 1

    fp = len(pred_instances) - tp
    fn = len(target_instances) - tp

    return tp, fp, fn


def average_precision(predictions, ground_truths, scores, iou_threshold=0.5):
    """
    Calcula AP para um IoU threshold.
 
    Args:
        predictions:
            lista de listas de instâncias previstas

        ground_truths:
            lista de listas de instâncias verdadeiras

        scores:
            lista de listas de scores

        iou_threshold:
            limiar de IoU

    Returns:
        AP
    """

    all_detections = []
    total_gt = 0

    for pred_instances, target_instances, pred_scores in zip(
        predictions,
        ground_truths,
        scores
    ):

        total_gt += len(target_instances)

        if len(pred_instances) == 0:
            continue

        order = np.argsort(-np.asarray(pred_scores))

        pred_instances = [pred_instances[i] for i in order]

        pred_scores = [pred_scores[i] for i in order]

        matched_targets = set()

        for pred, score in zip(pred_instances, pred_scores):

            best_iou = 0.0
            best_target = None

            for target_idx, target in enumerate(target_instances):

                if target_idx in matched_targets:
                    continue

                iou = instance_IoU(
                    pred,
                    target
                )

                if iou > best_iou:
                    best_iou = iou
                    best_target = target_idx

            if (best_target is not None and best_iou >= iou_threshold):
                matched_targets.add(best_target)
                is_tp = 1

            else:
                is_tp = 0

            all_detections.append(
                {
                    "score": score,
                    "tp": is_tp
                }
            )

    if total_gt == 0:
        return 0.0

    # Ordenar todas as previsões do dataset
    all_detections.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    tp = np.array([
        d["tp"]
        for d in all_detections
    ])

    fp = 1 - tp

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    precision = (
        cumulative_tp /
        np.maximum(cumulative_tp + cumulative_fp, 1)
    )

    recall = (
        cumulative_tp /
        total_gt
    )

    # Adiciona extremos
    recall = np.concatenate(
        ([0.0], recall, [1.0])
    )

    precision = np.concatenate(
        ([1.0], precision, [0.0])
    )

    # Precision envelope
    for i in range(
        len(precision) - 2,
        -1,
        -1
    ):
        precision[i] = max(
            precision[i],
            precision[i + 1]
        )

    # Área sob Precision-Recall
    indices = np.where(
        recall[1:] != recall[:-1]
    )[0]

    ap = np.sum(
        (
            recall[indices + 1]
            - recall[indices]
        )
        *
        precision[indices + 1]
    )

    return float(ap)


def mean_average_precision(predictions, ground_truths, scores, thresholds=np.arange(0.50, 0.951, 0.05)):
    """
    mAP para IoU thresholds de 0.50 até 0.95
    com passo de 0.05.
    """

    aps = {}

    for threshold in thresholds:

        ap = average_precision(
            predictions,
            ground_truths,
            scores,
            iou_threshold=threshold
        )

        aps[round(float(threshold), 2)] = ap

    map_value = np.mean(
        list(aps.values())
    )

    return map_value, aps


def count_tp_fp_fn(predictions, ground_truths, iou_threshold):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred_instances, gt_instances in zip(
        predictions,
        ground_truths
    ):

        tp, fp, fn = match_instances(
            pred_instances,
            gt_instances,
            iou_threshold
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

    return (
        total_tp,
        total_fp,
        total_fn
    )


def instance_scores(foreground_probability, instances):
    """
    Calcula uma confiança para cada instância.

    A confiança é a probabilidade média de foreground
    dentro da região da instância.
    """

    scores = []

    for instance in instances:
        if instance.sum() == 0:
            scores.append(0.0)
            continue

        score = foreground_probability[instance].mean()
        scores.append(float(score))

    return scores

def evaluate_image(pred_instances, target_instances, pred_scores, iou_threshold):
    """
    Avalia uma imagem.

    Retorna:
        lista de 1/0 indicando TP ou FP para
        cada previsão, ordenada por score.
    """

    if len(pred_instances) == 0:
        return [], len(target_instances)

    # Ordenar pelas maiores confianças
    order = np.argsort(-np.asarray(pred_scores))

    pred_instances = [pred_instances[i] for i in order]

    pred_scores = [pred_scores[i] for i in order]

    matched_targets = set()

    tp_flags = []

    for pred in pred_instances:

        best_iou = 0.0
        best_target = None

        for target_idx, target in enumerate(target_instances):

            if target_idx in matched_targets:
                continue

            iou = instance_IoU(pred, target)

            if iou > best_iou:
                best_iou = iou
                best_target = target_idx

        if (best_target is not None and best_iou >= iou_threshold):

            matched_targets.add(best_target)

            # True Positive
            tp_flags.append(1)

        else:

            # False Positive
            tp_flags.append(0)

    false_negatives = (len(target_instances) - len(matched_targets))

    return tp_flags, false_negatives