import torch.nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY

_reduction_modes = ['none', 'mean', 'sum']


@LOSS_REGISTRY.register()
class MarginRankingLoss(nn.Module):

    def __init__(self, loss_weight=1.0, margin=0.0, reduction='mean'):
        super(MarginRankingLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.margin = margin
        self.reduction = reduction

    def forward(self, input1, input2, target):
        return self.loss_weight * F.margin_ranking_loss(
            input1, input2, target, margin=self.margin, reduction=self.reduction)
