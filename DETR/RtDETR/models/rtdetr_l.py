import torch
import torch.nn as nn
from DETR.RtDETR.detr_layers import HGStem, HGBlock, DWConv, AIFI, Conv, RTDETRDecoder, Concat, RepC3

class RTDETR_L(nn.Module):
    def __init__(self, num_classes, scales):
        super().__init__()
        self.layer0 = HGStem(3, 32, 48)
        self.layer1 = HGBlock(48, 48, 128)
        self.layer2 = DWConv(128, 128, kernel_size=3, stride=2, dilation=1, act=False)
        self.layer3 = HGBlock(128, 96, 512, act=nn.ReLU())
        # self.layer3 = HGBlock(128, 96, 512, kernel_size=3, num_repeats=6, lightconv=False, shortcut=False, act=nn.ReLU())
        self.layer4 = DWConv(512, 512, kernel_size=3, stride=2, dilation=1, act=False)
        self.layer5 = HGBlock(512, 192, 1024, kernel_size=5, lightconv=True, act=nn.ReLU())
        self.layer6 = HGBlock(1024, 192, 1024, kernel_size=5, lightconv=True, act=nn.ReLU())
        self.layer7 = HGBlock(1024, 192, 1024, kernel_size=5, lightconv=True, act=nn.ReLU())
        self.layer8 = DWConv(1024, 1024, kernel_size=3, stride=2, dilation=1, act=False)
        self.layer9 = HGBlock(1024, 384, 2048, kernel_size=5, lightconv=True, act=nn.ReLU())
        self.layer10 = Conv(2048, 256)
        self.layer11 = AIFI(256, 1024)
        self.layer12 = Conv(256, 256)
        self.layer13 = nn.Upsample(scale_factor=2.0, mode='nearest')
        self.layer14 = Conv(1024, 256)
        self.layer15 = Concat()
        self.layer16 = RepC3(512, 256)
        self.layer17 = Conv(256, 256)
        self.layer18 = nn.Upsample(scale_factor=2.0, mode='nearest')
        self.layer19 = Conv(512, 256, act=False)
        self.layer20 = Concat()
        self.layer21 = RepC3(512, 256)
        self.layer22 = Conv(256, 256, kernel_size=3, stride=2)
        self.layer23 = Concat()
        self.layer24 = RepC3(512, 256)
        self.layer25 = Conv(256, 256, kernel_size=3, stride=2)
        self.layer26 = Concat()
        self.layer27 = RepC3(512, 256)

        self.decoder = None

    def forward(self, x):
        # backbone
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x = self.layer4(x3)
        x = self.layer5(x)
        x = self.layer6(x)
        x7 = self.layer7(x)
        x = self.layer8(x7)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x12 = self.layer12(x)
        x = self.layer13(x12)
        x14 = self.layer14(x7)
        x = self.layer15([x14, x])
        x = self.layer16(x)
        x17 = self.layer17(x)
        x18 = self.layer18(x)
        x19 = self.layer19(x3)
        x = self.layer20([x18, x19])

        # neck
        x21 = self.layer21(x)              # head1_S
        x = self.layer22(x21)
        x = self.layer23([x17, x])
        x24 = self.layer24(x)              # head2_M
        x = self.layer25(x24)
        x = self.layer26([x12, x])
        x27 = self.layer27(x)              # head3_L
        return [x21, x24, x27]

model = RTDETR_L(num_classes=80, scales={'l': [1.00, 1.00, 1024]})


class RTDETR_Decoder(nn.Module):
    def __init__(self,
                 nc=80,
                 ch=(256, 256, 256),
                 hd=256,   # hidden dim
                 ):
        super().__init__()
        self.nl = len(ch)  # num level
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False)) for x in ch)

    def forward(self, x):
        feats, shapes = self._get_encoder_input(x)

        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group()

        pass


    def _get_encoder_input(self, x):
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # b,c,h,w -> b,hw,c
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes



head = RTDETR_Decoder()

if __name__ == '__main__':
    # data = torch.randn([2, 3, 640, 640])
    data = torch.randn([2, 3, 224, 224])
    res_model = model(data)
    res_head = head(res_model)
    print(res_model[0].shape)   # torch.Size([2, 256, 80, 80])
    print(res_model[1].shape)   # torch.Size([2, 256, 40, 40])
    print(res_model[2].shape)   # torch.Size([2, 256, 20, 20])