import torch
import torch.nn as nn

loss_names = ['l1', 'l2']


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "MSE_Loss: inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss


class Distance(nn.Module):

    def __init__(self):
        super(Distance, self).__init__()

    def forward(self, student, teacher, gt):
        assert student.dim() == teacher.dim() and student.dim() == gt.dim(), "KL_DIV: inconsistent dimensions"
        valid_mask = (gt > 0).detach()
        valid_mask = valid_mask.expand(student.shape)
        diff = student - teacher
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):

    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
