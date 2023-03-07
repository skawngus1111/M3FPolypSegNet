import os
import random

import torch
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image

class PolypImageSegDataset(Dataset) :
    def __init__(self, dataset_dir, mode='train', transform=None, target_transform=None):
        super(PolypImageSegDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.image_folder = 'Original'
        self.label_folder = 'Ground Truth'
        self.transform = transform
        self.target_transform = target_transform
        self.frame = pd.read_csv(os.path.join(dataset_dir, '{}_frame.csv'.format(mode)))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if 'CVC-ClinicDB' in self.dataset_dir:
            image_path = os.path.join(self.dataset_dir, '/'.join(self.frame.png_image_path[idx].split('/')[1:]))
            label_path = os.path.join(self.dataset_dir, '/'.join(self.frame.png_mask_path[idx].split('/')[1:]))
        elif 'BKAI-IGH-NeoPolyp' in self.dataset_dir:
            image_path = os.path.join(self.dataset_dir, 'train', 'train', self.frame.path[idx])
            label_path = os.path.join(self.dataset_dir, 'train_gt', 'train_gt', self.frame.path[idx])

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform:
            seed = random.randint(0, 2 ** 32)
            self._set_seed(seed); image = self.transform(image)
            self._set_seed(seed); label = self.target_transform(label)
        label[label >= 0.5] = 1; label[label < 0.5] = 0

        return image, label

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)