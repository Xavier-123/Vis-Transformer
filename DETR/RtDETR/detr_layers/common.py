import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def autopad(k, p=None, d=1):
    '''Pad to 'same' shape outputs.'''
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c_in, c_out, kernel_size=1, stride=1, padding=None, g=1, d=1, act=True):
        """(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
        super(Conv, self).__init__()
        # Initialize the layers for Conv
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, autopad(kernel_size, padding, d), groups=g, dilation=d,
                              bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # Implement the forward pass for Conv
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class HGStem(nn.Module):
    '''StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.'''

    def __init__(self, c_in, c_mid, c_out):
        super().__init__()
        self.stem1 = Conv(c_in, c_mid, kernel_size=3, stride=2, act=nn.ReLU())
        self.stem2a = Conv(c_mid, c_mid // 2, kernel_size=2, stride=1, padding=0, act=nn.ReLU())
        self.stem2b = Conv(c_mid // 2, c_mid, kernel_size=2, stride=1, padding=0, act=nn.ReLU())
        self.stem3 = Conv(c_mid * 2, c_mid, kernel_size=3, stride=2, act=nn.ReLU())
        self.stem4 = Conv(c_mid, c_out, kernel_size=1, stride=1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])  # (左边填充数， 右边填充数， 上边填充数， 下边填充数)
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class DWConv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c_in, c_out, kernel_size=1, stride=1, padding=None, dilation=1,
                 act=True):  # # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__()
        # Initialize the layers for DWConv
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, autopad(kernel_size, padding, dilation),
                              groups=math.gcd(c_in, c_out), dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # Implement the forward pass for DWConv
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LightConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=1, act=nn.ReLU()):
        super().__init__()
        self.conv1 = Conv(c_in, c_out, kernel_size=1, act=False)
        self.conv2 = DWConv(c_out, c_out, kernel_size=kernel_size, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class HGBlock(nn.Module):
    def __init__(self, c_in, c_mid, c_out, kernel_size=3, num_repeats=6, lightconv=False, shortcut=False,
                 act=nn.ReLU()):
        super(HGBlock, self).__init__()
        '''
        HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
        '''
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(
            block(c_in if i == 0 else c_mid, c_mid, kernel_size=kernel_size, act=act) for i in range(num_repeats))
        self.sc = Conv(c_in + num_repeats * c_mid, c_out // 2, 1, 1, act=act)
        self.ec = Conv(c_out // 2, c_out, 1, 1, act=act)
        self.add = shortcut and c_out == c_in

    def forward(self, x):
        # Implement the forward pass for HGBlock
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = torch.cat(y, 1)
        y = self.sc(y)
        y = self.ec(y)
        return y + x if self.add else y


class TransformerEncoderLayer(nn.Module):
    def __init__(self, c_in, c_mid=2048, num_head=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.mulHeadAtt = nn.MultiheadAttention(c_in, num_head, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(c_in, c_mid)
        self.fc2 = nn.Linear(c_mid, c_in)

        self.norm1 = nn.LayerNorm(c_in)
        self.norm2 = nn.LayerNorm(c_in)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, src, src_mask=None, src_key_padding_mak=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.mulHeadAtt(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mak)[0]
        src += self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.mulHeadAtt(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    def __init__(self, c_in, c_mid=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        super().__init__(c_in, c_mid, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # flatten [B, C, H, W] to [B, HxW, C]
        x = x.flatten(2).permute(0, 2, 1)
        x = super().forward(x, pos=pos_embed.to(device=x.device, dtype=x.dtype))
        x = x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()   # 还原
        return x

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], axis=1)[None, :, :]

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        # Initialize the layers for Concat
        self.d = dimension

    def forward(self, x):
        # Implement the forward pass for Concat
        return torch.cat(x, self.d)


class RepConv(nn.Module):
    '''RepConv is a basic rep-style block, including training and deploy status'''
    default_act = nn.SiLU()
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1, group=1, dilation=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert kernel_size == 3 and padding == 1
        self.g = group
        self.c_in = c_in
        self.c_out = c_out
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c_in) if bn and c_out == c_in and stride == 1 else None
        self.conv1 = Conv(c_in, c_out, kernel_size, stride, padding=padding, g=group, act=False)
        self.conv2 = Conv(c_in, c_out, 1, stride, padding=(padding - kernel_size // 2), g=group, act=False)

    def forward(self, x):
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    # 推理用？
    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))


class RepC3(nn.Module):
    def __init__(self, c_in, c_out, n=3, e=1.0):
        super().__init__()
        '''
        RepC3 implementation.
        '''
        # Initialize the layers for RepC3
        c_ = int(c_out * e)
        self.cv1 = Conv(c_in, c_out, 1, 1)
        self.cv2 = Conv(c_in, c_out, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c_out, 1, 1) if c_ != c_out else nn.Identity()

    def forward(self, x):
        # Implement the forward pass for RepC3 layer
        x1 = self.cv1(x)
        x1 = self.m(x1)
        x2 = self.cv2(x)
        x = self.cv3(x2 + x1)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 attn_dropout=None, act_dropout=None, normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, pos_embed=None, query_pos_embed=None):
        tgt_mask = torch.tensor(tgt_mask, dtype=torch.bool) if tgt_mask is not None else None

        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        tgt = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)[0]
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        q = self.with_pos_embed(tgt, query_pos_embed)
        k = self.with_pos_embed(memory, pos_embed)
        tgt = self.cross_attn(q, k, value=memory, attn_mask=memory_mask)[0]
        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, pos_embed=None, query_pos_embed=None):
        tgt_mask = torch.tensor(tgt_mask, dtype=torch.bool) if tgt_mask is not None else None

        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           pos_embed=pos_embed, query_pos_embed=query_pos_embed)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class RTDETRDecoder(nn.Module):
    def __init__(self, num_classes):
        super(RTDETRDecoder, self).__init__()
        '''
        RTDETR: Decoder implementation.

        params:
                num_classes (int): Number of classes in the dataset.
        '''
        # Initialize the layers for RTDETRDecoder
        self.num_classes = num_classes
        d_model = 512  # Set the appropriate value for d_model
        nhead = 8  # Set the appropriate value for nhead
        dim_feedforward = 2048  # Set the appropriate value for dim_feedforward
        dropout = 0.1  # Set the appropriate value for dropout
        activation = "relu"  # Set the appropriate value for activation

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None, pos_embed=None, query_pos_embed=None):
        # Implement the forward pass for RTDETRDecoder
        output = self.transformer_decoder(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                          pos_embed=pos_embed, query_pos_embed=query_pos_embed)
        return output


if __name__ == '__main__':
    data = torch.randn([1, 3, 640, 640])
    model = HGStem(3, 16, 64)
    data = model(data)
    print(data.shape)
