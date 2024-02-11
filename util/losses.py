import torch
import torch.nn as nn

def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims

class Loss(nn.Module):
    def __init__(self, pred_outputs, gt_outputs):
        super().__init__()
        self.pred_outputs = pred_outputs
        self.gt_outputs = gt_outputs


class MSE(Loss):
    def __init__(self, pred_name='images', gt_image_name='target_images'):
        super(MSE, self).__init__(pred_outputs=(pred_name,), gt_outputs=(gt_image_name,))

    def forward(self, pred, label):
        label = label.view(pred.size())
        loss = torch.mean((pred - label) ** 2, dim=get_dims_with_exclusion(label.dim(), 0))
        return loss


class TVLoss(nn.Module):
    def __init__(self, strength):
        super(TVLoss, self).__init__()
        self.strength = strength
        self.x_diff = torch.Tensor()
        self.y_diff = torch.Tensor()

    def forward(self, input):
        self.x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        self.y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        self.loss = self.strength * (torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff)))
        # return input
        return self.loss
