import torch
import torch.nn as nn


class SetCriterion(nn.Module):
    ''' 计算DETR的loss（1、计算匈牙利匹配；2、计算预测值与目标值差异） '''

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        # eos_cof:应用于无对象类别的相对分类权重
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self):
        return

    def loss_boxes(self):
        return

    def loss_cardinality(self):
        return

    def loss_mask(self):
        return

    def get_loss(self):
        return

    def forward(self):
        pass
