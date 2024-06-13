import numpy as np
import pandas as pd
import h5py
import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import random
import xarray as xr

from .data_utils import *
from torch.utils.data import Dataset

class CropTypeDataset(Dataset):
    def __init__(
        self,
        data_path,
        keys: dict = None,
        bands=['vhvv', 'vh', 'vv'],
        transform=None
        ):
        
        self.path = data_path
        self.bands = bands
        self.keys = keys
        
        with h5py.File(self.path, 'r') as f:
            self.len = len(f['label'])
            if self.keys is None:
                labels = np.unique(f['label'])
                self.keys = encode_labels(labels)
        
        self.transform = transform
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        series_data = []
        with h5py.File(self.path, 'r') as f:
            for band in self.bands:    
                series_data.append(f[band][index]) # Retrieve ts data for each band
            label = f['label'][index]
            
        # One hot encode label
        encoded = np.zeros(len(self.keys.keys())) 
        encoded[self.keys[label]] = 1
        
        # Combine labels and data
        signals = np.vstack((
            series_data
            ))
        
        return signals, encoded

    def convert_label(self, encoded_vec):
        label = list(self.keys.keys())[int(np.where(encoded_vec == 1)[0].item())]
        return label
    
    def get_ds(self):
        return xr.open_dataset(os.path.join(self.path))
