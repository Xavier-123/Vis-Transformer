import torch
import yaml
from utils.load import DEFAULT_CFG

from DETR.DETR.models.detr import build

def train(args):
    device = torch.device(args.device)

    # 创建模型
    model = build(args)

    # 数据阶段

    # 创建优化器

    # 开始训练

    # 保存模型

def main(args):
    train(args)

if __name__ == '__main__':
    main(DEFAULT_CFG)

