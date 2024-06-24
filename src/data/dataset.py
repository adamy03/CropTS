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
        data,
        labels,
        bands=None,
        seq_len=None,
        transform=None
    ):
        self.bands = bands
        self.seq_len = seq_len
        self.transform = transform
        self.data = data
        self.labels = labels
        
        self.keys = encode_labels(labels)
        self.len = data.shape[0]
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        if self.bands:
            data = self.data[index, self.bands, :]
        else:
            data = self.data[index]
        
        if self.seq_len:
            data = data[:, :self.seq_len]
            
        label = self.labels[index]

        if self.transform:
            data = self.transform(data)
            label = self.transform(label)
                        
        return data, label

    def convert_label(self, encoded_vec):
        label = list(self.keys.keys())[int(np.where(encoded_vec == 1)[0].item())]
        
        return label
    
    def get_keys(self, reversed=False):
        if reversed:
            return {v: k for k, v in self.keys.items()}
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
    
    