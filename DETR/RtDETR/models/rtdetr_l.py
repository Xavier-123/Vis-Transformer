import torch
import torch.nn as nn
from DETR.RtDETR.detr_layers import HGStem, HGBlock, DWConv, AIFI, Conv, RTDETRDecoder

class RTDETR_L(nn.Module):
    def __init__(self, num_classes, scales):
        super().__init__()


    def forward(self, x):
        pass

model = RTDETR_L(num_classes=80, scales={'l': [1.00, 1.00, 1024]})