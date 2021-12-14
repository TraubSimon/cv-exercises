import torch
import torch.nn as nn
import torch.nn.functional as F


from .metrics import aepe, pointwise_epe


class FlowLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.weight_decay = 1e-4
        self.gt_interpolation = 'bilinear'

        self.loss_weights = [1 / 16, 1 / 16, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1]
        self.loss_weights = [100 * weight for weight in self.loss_weights]
        self.reg_params = self.get_regularization_parameters(model)

    def get_regularization_parameters(self, model):
        reg_params = []
        for name, param in model.named_parameters():
            if 'pred' not in name and not name.endswith('bias') and not name.endswith('bn.weight') and param.requires_grad:
                reg_params.append((name, param))

        print("Applying regularization loss with weight decay {} on:".format(self.weight_decay))
        for i, val in enumerate(reg_params):
            name, param = val
            print("\t#{} {}: {} ({})".format(i, name, param.shape, param.numel()))
        print()

        return reg_params

    def forward(self, sample, model_output):

        pointwise_losses = {}
        sub_losses = {}

        gt_flow = sample['gt_flow']
        pred_flows_all = model_output['pred_flows_all']

        total_aepe_loss = 0
        total_reg_loss = 0

        for level, pred_flow in enumerate(pred_flows_all):

            with torch.no_grad():
                gt_flow_resampled = F.interpolate(gt_flow, size=pred_flow.shape[-2:], mode=self.gt_interpolation,
                                                  align_corners=(False if self.gt_interpolation != 'nearest' else None))

            aepe_loss = aepe(gt=gt_flow_resampled, pred=pred_flow, weight=self.loss_weights[level])
            pointwise_epe_ = pointwise_epe(gt=gt_flow_resampled, pred=pred_flow, weight=self.loss_weights[level])

            sub_losses['0_aepe/level_%d' % level] = aepe_loss
            pointwise_losses['1_epe/level_%d' % level] = pointwise_epe_

            total_aepe_loss += aepe_loss

        for name, param in self.reg_params:
            reg_loss = torch.sum(torch.mul(param, param)) / 2.0
            total_reg_loss += reg_loss

        total_reg_loss *= self.weight_decay

        total_loss = total_aepe_loss + total_reg_loss

        sub_losses['1_total_aepe'] = total_aepe_loss
        sub_losses['2_reg'] = total_reg_loss

        return total_loss, sub_losses, pointwise_losses


class PhotometricLoss(nn.Module):
    def __init__(self, model, use_smoothness_loss=True):
        super().__init__()
        

    def forward(self, sample, model_output):
        # TODO: implement the forward loss function of the photometric loss
        return
