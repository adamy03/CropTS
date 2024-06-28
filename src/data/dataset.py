import numpy as np
import pandas as pd
import h5py
import os
import sys
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import random

from .data_utils import *
from torch.utils.data import Dataset

class CropTypeDataset(Dataset):
    def __init__(
        self,
        path,
        subset='train',
        bands=None,
        seq_len=None,
        transform=None,
        include_masks=False
    ):
        self.seq_len = seq_len
        self.transform = transform
        self.bands = bands
        self.include_masks = include_masks
        
        if not subset in ['train', 'val', 'test']:
            raise ValueError('subset must be "train", "val", or "test"')

        self.data = np.load(os.path.join(path, f'{subset}_signals.npy'), allow_pickle=True)
        
        if include_masks:
            self.masks = np.load(os.path.join(path, f'{subset}_masks.npy'), allow_pickle=True)
        
        self.labels = np.load(os.path.join(path, f'{subset}_labels.npy'), allow_pickle=True)
        self.no_unique_labels = len(np.unique(self.labels))
        self.keys = dict(zip([i for i in np.unique(self.labels)], [np.identity(self.no_unique_labels)[i,:] for i in range(self.no_unique_labels)]))
        self.labels = np.array([self.keys[label] for label in self.labels])
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        if self.bands:
            data = self.data[index, self.bands, :]
        else:
            data = self.data[index, :, :]
        
        if self.seq_len:
            data = data[:, :self.seq_len]
        
        data = torch.from_numpy(data.astype(np.float32))
        label = torch.from_numpy(self.labels[index])
        
        if self.transform:
            data = self.transform(data)
            label = self.transform(label)
        
        if self.include_masks: 
            mask = torch.from_numpy(self.masks[index])
            return data, label, mask
        else:                    
            return data, label

    def convert_label(self, encoded_vec):
        label = list(self.keys.keys())[int(np.where(encoded_vec == 1)[0].item())]
        
        return label
    
    def get_keys(self, reversed=False):
        if reversed:
            return {np.argmax(v): k for k, v in self.keys.items()}
        else:
            return self.keys
    
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return torch.Tensor(self.embeddings[index]), torch.Tensor(self.labels[index])
    
    