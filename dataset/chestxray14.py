# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd


data_folder = './data/chestxray14/'


class ChestXray14Dataset(Dataset):
    """
    ChestX-ray14 dataset
    """
    def __init__(self, csv_file, img_dir, transform=None, train=True):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.train = train
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        # Get labels (14 classes for ChestX-ray14)
        labels = self.data_frame.iloc[idx, 1:15].values.astype(np.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels, idx


class ChestXray14Instance(ChestXray14Dataset):
    """
    ChestX-ray14 instance dataset
    """
    def __getitem__(self, index):
        img, target, idx = super().__getitem__(index)
        return img, target, idx


def get_chestxray14_dataloaders(batch_size=128, num_workers=8, is_instance=False):
    """
    Get ChestX-ray14 data loaders
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_csv = os.path.join(data_folder, 'train_labels.csv')
    test_csv = os.path.join(data_folder, 'test_labels.csv')
    img_dir = os.path.join(data_folder, 'images')

    if is_instance:
        train_set = ChestXray14Instance(csv_file=train_csv, img_dir=img_dir, transform=train_transform, train=True)
        n_data = len(train_set)
    else:
        train_set = ChestXray14Dataset(csv_file=train_csv, img_dir=img_dir, transform=train_transform, train=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    test_set = ChestXray14Dataset(csv_file=test_csv, img_dir=img_dir, transform=test_transform, train=False)
    test_loader = DataLoader(test_set, batch_size=int(batch_size/2), shuffle=False, num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader


class ChestXray14InstanceSample(ChestXray14Dataset):
    """
    ChestXray14Instance+Sample Dataset
    """
    def __init__(self, csv_file, img_dir, transform=None, train=True, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(csv_file=csv_file, img_dir=img_dir, transform=transform, train=train)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 14  # ChestX-ray14 has 14 classes

        num_samples = len(self.data_frame)
        
        # Get labels for all samples
        labels = []
        for i in range(num_samples):
            sample_labels = self.data_frame.iloc[i, 1:15].values
            # Convert to single label (use the first positive label or 0 if all negative)
            label = np.argmax(sample_labels) if np.any(sample_labels > 0) else 0
            labels.append(label)

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[labels[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target, idx = super().__getitem__(index)
    
        if not self.is_sample:
            return img, target, idx
        else:
            # Convert multi-label to single label for sampling
            single_target = np.argmax(target) if np.any(target > 0) else 0
            
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[single_target], 1)[0]
            else:
                raise NotImplementedError(self.mode)
    
            replace = True if self.k > len(self.cls_negative[single_target]) else False
            neg_idx = np.random.choice(self.cls_negative[single_target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, idx, sample_idx


def get_chestxray14_dataloaders_sample(batch_size=128, num_workers=1, k=4096, mode='exact', is_sample=True, percent=1.0):
    """
    Get ChestX-ray14 sample data loaders
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_csv = os.path.join(data_folder, 'train_labels.csv')
    test_csv = os.path.join(data_folder, 'test_labels.csv')
    img_dir = os.path.join(data_folder, 'images')

    train_set = ChestXray14InstanceSample(csv_file=train_csv, img_dir=img_dir, transform=train_transform, 
                                         train=True, k=k, mode=mode, is_sample=is_sample, percent=percent)
    n_data = len(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = ChestXray14Dataset(csv_file=test_csv, img_dir=img_dir, transform=test_transform, train=False)
    test_loader = DataLoader(test_set, batch_size=int(batch_size/2), shuffle=False, num_workers=int(num_workers/2))

    return train_loader, test_loader, n_data 