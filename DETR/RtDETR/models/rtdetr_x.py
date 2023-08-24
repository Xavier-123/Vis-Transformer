import torch
import torch.nn as nn
from DETR.RtDETR.detr_layers import HGStem, HGBlock, DWConv, AIFI, Conv, RTDETRDecoder


class RTDETR_X(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


model = RTDETR_X()