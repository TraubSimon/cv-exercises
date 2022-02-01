import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import warp, shift_multi


def aepe(gt, pred, mask=None, weight=None, eps=1e-9):

    epe = torch.linalg.norm(pred-gt, dim=1)

    if weight is not None:

        if isinstance(weight, torch.Tensor):
            while weight.ndim < epe.ndim:
                weight = weight.unsqueeze(-1)

        epe = epe * weight

    if mask is None:
        aepe = epe.mean()
    else:
        mask = mask.float()
        num_valid = torch.sum(mask)
        aepe = 1 / (num_valid + eps) * torch.sum(epe * mask)
        aepe = aepe * float((num_valid != 0))

    return aepe


def pointwise_epe(gt, pred, mask=None, weight=None):

    pointwise_epe = torch.linalg.norm(pred-gt, dim=1, keepdim=True)

    if mask is not None:
        pointwise_epe *= mask.float()

    if weight is not None:

        if isinstance(weight, torch.Tensor):
            while weight.ndim < pointwise_epe.ndim:
                weight = weight.unsqueeze(1)

        pointwise_epe *= weight

    return pointwise_epe


def photometric_loss(image_1, image_2, pred_flow, weight=None, eps=1e-3):

    pred_flow = F.interpolate(pred_flow, size=image_1.shape[-2:], mode='bilinear', align_corners=False)
    image_2_warped, warping_mask = warp(image_2, offset=pred_flow, padding_mode='zeros')
    photo_error = torch.abs(image_1 - image_2_warped)
    pointwise_photo_loss = ((photo_error**2 + eps**2)**0.5) * warping_mask

    if weight is not None:

        if isinstance(weight, torch.Tensor):
            while weight.ndim < pointwise_photo_loss.ndim:
                weight = weight.unsqueeze(-1)

        pointwise_photo_loss = pointwise_photo_loss * weight

    warping_mask = warping_mask.float()
    num_valid = torch.sum(warping_mask)
    photo_loss = 1 / (num_valid + eps) * torch.sum(pointwise_photo_loss)
    photo_loss = photo_loss * float((num_valid != 0))

    return image_2_warped, warping_mask, pointwise_photo_loss, photo_loss


def smoothness_loss(pred_flow, weight=None, eps=1e-3):

    offsets = torch.tensor([[1, 0], [0, 1]])
    shifteds, shift_masks = shift_multi(pred_flow, offsets)  # N22HW, NSHW
    smoothness_errors = pred_flow.unsqueeze(1) - shifteds  # N22HW
    pointwise_smoothness_loss = torch.sum(torch.sum((smoothness_errors**2 + eps**2)**0.3, dim=2), dim=1, keepdim=True)  # N1HW
    mask = (torch.sum(shift_masks, dim=1, keepdim=True) == len(offsets))  # N1HW
    pointwise_smoothness_loss = pointwise_smoothness_loss * mask

    if weight is not None:
        if isinstance(weight, torch.Tensor):
            while weight.ndim < pointwise_smoothness_loss.ndim:
                weight = weight.unsqueeze(-1)

        pointwise_smoothness_loss = pointwise_smoothness_loss * weight

    mask = mask.float()
    num_valid = torch.sum(mask)
    smoothness_loss = 1 / (num_valid + eps) * torch.sum(pointwise_smoothness_loss)
    smoothness_loss = smoothness_loss * float((num_valid != 0))

    return pointwise_smoothness_loss, smoothness_loss


def compute_flow_metrics(sample, model_output):
    image = sample['images'][0]
    gt_flow = sample['gt_flow']
    pred_flow = model_output['pred_flow']

    orig_ht, orig_wd = gt_flow.shape[-2:]
    pred_ht, pred_wd = image.shape[-2:]
    scale_ht, scale_wd = orig_ht/pred_ht, orig_wd/pred_wd

    pred_flow = F.interpolate(pred_flow, size=gt_flow.shape[-2:], mode='nearest')
    pred_flow[:, 0, :, :] = pred_flow[:, 0, :, :] * scale_wd
    pred_flow[:, 1, :, :] = pred_flow[:, 1, :, :] * scale_ht

    aepe_ = aepe(gt=gt_flow, pred=pred_flow).item()
    pointwise_epe_ = pointwise_epe(gt=gt_flow, pred=pred_flow)

    metrics = {
        'aepe': aepe_,
    }

    qualitatives = {
        'pred_flow': pred_flow,
        'epe': pointwise_epe_,
    }
    return metrics, qualitatives
