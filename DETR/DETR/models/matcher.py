import torch
import torch.nn as nn


class HungarianMatcher(nn.Module):
    '''匈牙利匹配，因此通常，预测多于目标。'''
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        return


def build_matcher(args):
    return HungarianMatcher(args.cost_class, args.cost_bbox, args.cost_giou)