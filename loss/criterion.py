import torch
import torch.nn as nn

class ModelCriterion(nn.Module):
    def __init__(self):
        super(ModelCriterion, self).__init__()

    def forward(self, inputY, target, mask):
        inputY = inputY.contiguous().view(-1, inputY.shape[2])
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous.view(-1, 1)

        output = -inputY.gather(1, target) * mask
        return torch.sum(output) / torch.sum(mask)