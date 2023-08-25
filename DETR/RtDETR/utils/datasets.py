import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision

import cv2
from PIL import Image
from pathlib import Path
import glob
import os
import numpy as np
import copy

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # image suffixes


class Label_Object():
    def __init__(self, labels, bboxs, num):
        self.labels = labels
        self.bboxs = bboxs
        self.num = num


# class RTDETR_Dataset(Dataset):
class RTDETR_Dataset(Dataset):
    def __init__(self, img_paths, imgsz=640, augment=True, stride=32, pad=0.5, **kwargs):
        super().__init__()
        self.img_paths = img_paths + "/images/train2017"
        self.img_list = os.listdir(self.img_paths)
        self.label_paths = img_paths + "/labels/train2017"
        self.label_lists = os.listdir(self.label_paths)
        self.transform = transforms.Compose([
            # transforms.Resize((416, 416)),
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        path = self.img_paths + "/" + self.img_list[index]
        img_pil = Image.open(path)
        h, w = img_pil.size
        img_pil = img_pil.resize((224, 224))
        img = self.transform(img_pil)

        return img, (h, w), self.label_paths + "/" + self.label_lists[index]

    def __len__(self):
        return len(self.img_paths)


def load_label(img_ori, label_paths, mode='coco'):
    label, bboxs, nums = [], [], []
    _data = []
    for i in range(len(label_paths)):
        h, w = img_ori[i]
        with open(label_paths[i], mode='r') as f:
            data = f.readlines()
            data = [i.replace("\n", "").split() for i in data]
            nums.append(len(data))
        for i in range(len(data)):
            _data.append([int(data[i][0]), float(data[i][1]), float(data[i][2]), float(data[i][3]), float(data[i][4])])

    if mode == 'coco':
        for j in _data:
            label.append(int(j[0]))
            x1, y1, x2, y2 = float(j[1] * h) - float(j[3] * h) / 2, float(j[2] * w) - float(j[4] * w) / 2, \
                             float(j[1] * h) + float(j[3] * h) / 2, float(j[2] * w) + float(j[4] * w) / 2
            bboxs.append([x1, y1, x2, y2])

    return label, bboxs, nums


if __name__ == '__main__':
    ds = RTDETR_Dataset(r"D:\Inspur\base_data\coco128\coco2017")
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    for idx, (im, io, la) in enumerate(dl):
        label, bboxs, nums = load_label(io, la)
        print(im.size())
        print(label)
        print(bboxs)
        print(nums)
        break
