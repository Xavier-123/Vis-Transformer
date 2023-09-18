from typing import Iterable
from pathlib import Path

import torch
import torch.utils.data
from torch.utils.data import DataLoader, DistributedSampler

from DETR.DETR.utils.misc import collate_fn
from DETR.DETR.utils.load import DEFAULT_CFG
from DETR.DETR.models.detr import build_model
from DETR.DETR.datasets import build_dataset


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()

    # for samples, targets in me


def train(args):
    device = torch.device(args.device)

    # 创建模型, 创建优化器, 数据处理
    model, criterion, postprocessors = build_model(args)
    # print(model)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('模型总共参数量:', n_parameters)



    # 优化器
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # dataset
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn,
                                 num_workers=args.num_workers)

    # 开始训练
    output_dir = Path(args.output_dir)
    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            pass
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()
        if args.output_dir:
            pass

    # 保存模型

def main(args):
    train(args)

if __name__ == '__main__':
    main(DEFAULT_CFG)

