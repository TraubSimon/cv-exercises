import torch
import torch.nn as nn
import torch.nn.functional as F


def aepe(gt, pred, weight=None, eps=1e-9):

    '''
    TODO:
    Implement the average end-point error function between the ground truth flow
    and the prediction flow
    hint: check the torch.linalg.norm() to compute the norm of a vector.
    weight: is a scalar value which is used to scale the metric. This is needed for training but not for testing. 
    '''
    # TODO: Implement the average end-point error function between

    return 


def pointwise_epe(gt, pred, weight=None):

    '''
    TODO:
    Implement the end-point error function between the ground truth flow
    and the prediction flow
    hint: check the torch.linalg.norm() to compute the norm of a vector.
    weight: is a scalar value which is used to scale the metric. This is needed for training but not for testing. 
    The main difference to the previous function is that here we do not average and the result is an error map.
    '''

    return


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
