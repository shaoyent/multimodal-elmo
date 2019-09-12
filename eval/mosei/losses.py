import torch
from torch.nn import Module
import torch.nn.functional as F

def weighted_binary_cross_entropy(sigmoid_x, targets, pos_weight=None, neg_weight=None, class_weight=None, size_average=True, reduce=True):
    """
    Args:
        sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
        targets: true value, one-hot-like vector of size [N,C]
        pos_weight: Weight for postive sample
    """
    if not (targets.size() == sigmoid_x.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

    loss = F.binary_cross_entropy_with_logits(sigmoid_x, targets, reduction='none')
    loss = pos_weight * targets * loss + \
           neg_weight * (1-targets) * loss

    if class_weight is not None:
        loss = loss * class_weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class WeightedBCELoss(Module):
    def __init__(self, pos_weight=None, neg_weight=None, class_weight=None, PosWeightIsDynamic= False, WeightIsDynamic= False, size_average=True, reduce=True, gamma=1):
        """
        Args:
            pos_weight = Weight for postive samples. Size [1,C]
            weight = Weight for Each class. Size [1,C]
            PosWeightIsDynamic: If True, the pos_weight is computed on each batch. If pos_weight is None, then it remains None.
            WeightIsDynamic: If True, the weight is computed on each batch. If weight is None, then it remains None.
        """
        super().__init__()

        self.register_buffer('weight', class_weight)
        self.register_buffer('gamma', gamma)
        # self.register_buffer('pos_weight', torch.pow(pos_weight, gamma))
        self.register_buffer('pos_weight', pos_weight)
        self.register_buffer('neg_weight', neg_weight)
        self.size_average = size_average
        self.reduce = reduce
        self.PosWeightIsDynamic = PosWeightIsDynamic

    def forward(self, input, target):
        # pos_weight = Variable(self.pos_weight) if not isinstance(self.pos_weight, Variable) else self.pos_weight
        if self.PosWeightIsDynamic:
            nBatch = len(target)
            positive_counts = target.sum(dim=0)
            negative_counts = -positive_counts + nBatch

            # self.pos_weight = negative_counts / (positive_counts +1e-5)
            # self.neg_weight = 1 + positive_counts / nBatch

            # s = self.pos_weight + self.neg_weight
            # self.pos_weight /= s
            # self.neg_weight /= s

            self.pos_weight = negative_counts / (positive_counts +1e-2)
            self.neg_weight = self.gamma

            # weight = Variable(self.weight) if not isinstance(self.weight, Variable) else self.weight
        return weighted_binary_cross_entropy(input, target,
                                             pos_weight=self.pos_weight,
                                             neg_weight=self.neg_weight,
                                             class_weight=self.weight,
                                             size_average=self.size_average,
                                             reduce=self.reduce)
