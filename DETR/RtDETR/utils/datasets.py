import torch
import torch.nn as nn
from torch.utils.data import Dataset


class RTDETR_Dataset(Dataset):
    def __init__(self, *args, data=None, **kwargs):
        super().__init__(*args, data=data, use_segments=False, use_keypoints=False, **kwargs)
        self.data = self.load_data()

    def __getitem__(self, index):
        img, target = self.data[index]
        return

    def __len__(self):
        return

    def load_data(self):
        return