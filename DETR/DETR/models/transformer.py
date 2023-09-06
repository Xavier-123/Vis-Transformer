import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Optional
from torch import Tensor
from DETR.DETR.utils.misc import NestedTensor


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class PositionEmbeddingSine(nn.Module):
    '''固定的位置编码'''

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:, :] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    '''可学习的位置编码'''

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")
    return position_embedding


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        # 编码器
        encoder_layer = TransformerEncoderLayer(d_model,
                                                nhead,
                                                dim_feedforward,
                                                dropout,
                                                activation,
                                                normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # 解码器
        decoder_layer = TransformerDecoderLayer(d_model,
                                                nhead,
                                                dim_feedforward,
                                                dropout,
                                                activation,
                                                normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor] = None):
        if pos:
            return tensor + pos
        return tensor

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        _src = self.norm1(src)
        q = k = self.with_pos_embed(_src, pos)
        _src = self.self_attn(q, k, value=_src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(_src)
        src = self.norm2(src)
        _src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(_src)
        return src

    def forward_post(self, src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        _src = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(_src)
        src = self.norm1(src)
        _src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(_src)
        src = self.norm2(src)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        '''
        1、key_padding_mask用于屏蔽掉被填充的位置，即，输入序列的结束之后。这始终特定于输入批次，并取决于批次中的序列与最长的序列相比有多长。它是一个形状为批量大小×输入长度的二维张量。
        2、attn_mask说明哪些键值对是有效的。在 Transformer 解码器中，三角形掩码用于模拟推理时间并防止关注“未来”位置。这是att_mask通常使用的。如果是二维张量，
        则形状为input length × input length。您还可以拥有一个特定于批次中每个项目的掩码。在这种情况下，您可以使用形状为(batch size × num head) × input length × input length的 3D 张量。
        （因此，理论上，您可以key_padding_mask使用 3D进行模拟att_mask。）
        3、attn_mask只用于Decoder训练时的解码过程，作用是掩盖掉当前时刻之后的信息，让模型只能看到当前时刻（包括）之前的信息。
        '''
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask, src_key_padding_mask, pos):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask, src_key_padding_mask, pos)

        if self.norm is not None:
            out = self.norm(out)
        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, tgt, memory, tgt_mask, memory_mask,
                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos):
        _tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(_tgt, query_pos)
        _tgt = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        _tgt = _tgt[0]
        tgt = tgt + self.dropout1(_tgt)
        _tgt = self.norm2(tgt)
        _tgt = self.multihead_attn(query=self.with_pos_embed(_tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        _tgt = _tgt[0]
        tgt = tgt + self.dropout2(_tgt)
        _tgt = self.norm3(tgt)
        _tgt = self.linear2(self.dropout(self.activation(self.linear1(_tgt))))
        tgt = tgt + self.dropout3(_tgt)
        return tgt

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     ):
        q = k = self.with_pos_embed(tgt, query_pos)
        _tgt = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(_tgt)
        tgt = self.norm1(tgt)
        _tgt = self.multihead_attn(query_pos=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(_tgt)
        _tgt = self.linear2(self.dropout(self.dropout(self.activation(self.linear1(tgt)))))
        tgt = tgt + self.dropout3(_tgt)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask,
                                 memory_key_padding_mask, pos, query_pos)


class TransformerDecoder(nn.Module):
    # return_intermediate 是否返回中间层
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


def build_transformer(args):
    model = Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )

    # model = Transformer(
    #     d_model=256,
    #     dropout=0.1,
    #     nhead=8,
    #     dim_feedforward=2048,
    #     num_encoder_layers=2,
    #     num_decoder_layers=1,
    #     normalize_before=True,
    #     return_intermediate_dec=True,
    # )
    return model


if __name__ == '__main__':

    # samples = NestedTensor(torch.rand([2, 3, 672, 902]), torch.rand([2, 672, 902]))
    # target = [
    #     {"boxes": torch.tensor([100, 50, 20, 10]).to("cuda:0")},
    #     {"labels": torch.tensor([1]).to("cuda:0")},
    #     {"image_id": torch.tensor([1]).to("cuda:0")},
    #     {"area": torch.tensor([150]).to("cuda:0")},
    #     {"iscrowd": torch.tensor([0]).to("cuda:0")},
    #     {"orig_size": torch.tensor([480, 640]).to("cuda:0")},
    #     {"size": torch.tensor([640, 853]).to("cuda:0")},
    # ]
    # # mask = torch.rand([2, 672, 902]).to(torch.bool)
    # model = build_transformer(1)
    # print(model)
    # # src, mask, query_embed, pos_embed
    # query_embed = torch.rand([2, 256, 21, 29]).to("cuda:0")
    # pos_embed = PositionEmbeddingSine(256//2, normalize=True)
    # x = model(samples)
    # print(x.shape)
    # print()

    d_model = 512
    input_data = torch.rand([1, 3, 640, 640])
    import torchvision
    from torchvision.models._utils import IntermediateLayerGetter
    # 1、backbone
    # backbone = getattr(torchvision.models, 'resnet18')(
    backbone = getattr(torchvision.models, 'resnet34')(
            replace_stride_with_dilation=[False, False, False],
            pretrained=True,
            # norm_layer=FrozenBatchNorm2d
    )
    body = IntermediateLayerGetter(backbone, return_layers={'layer4': "0"})
    print(input_data.shape)
    x = body(input_data)    # NestedTensor
    print(x.shape)

    # 2、pos
    pos = PositionEmbeddingSine(d_model // 2, normalize=True)
    x = pos(x)

    # 3、encoder
    encoder = None

    # 4、decoder
    decoder = None

    # loss
    # loss_dict = criterion(outputs, targets)