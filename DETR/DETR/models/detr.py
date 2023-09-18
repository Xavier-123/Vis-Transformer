import torch
import torch.nn as nn
import torch.nn.functional as F

from DETR.DETR.utils.misc import NestedTensor
from DETR.DETR.models.backbone import build_backbone
from DETR.DETR.models.transformer import build_transformer
from DETR.DETR.models.matcher import build_matcher
from DETR.DETR.models.criterion import SetCriterion
from DETR.DETR.models.postprocessors import PostProcess


class MLP(nn.Module):
    '''FFN'''
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers -1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            pass
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()   # 取-1是因为features保存了resnet每层的结果，取-1代表我只取最后一层的结果。
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused     # @torch.jit.unused 允许您在模型中保留与 TorchScript 不兼容的代码，但仍可以导出模型（即训练用，推理不用）
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

def build_model(args):

    num_classes = args.num_classes
    device = torch.device(args.device)

    # backbone
    backbone = build_backbone(args)

    # transformer
    transformer = build_transformer(args)

    # model
    model = DETR(backbone, transformer, num_classes, args.num_queries)
    model.to(device)

    # loss
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    # 计算辅助头loss
    if args.aux_loss:
        pass

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ['masks']
    criterion = SetCriterion(num_classes=num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)

    # postprocessors
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        pass

    return model, criterion, postprocessors
