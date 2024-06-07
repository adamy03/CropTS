import numpy as np
import pandas as pd
import h5py
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import random
import xarray as xr

from torch.utils.data import Dataset

class CropTypeDataset(Dataset):
    def __init__(
        self,
        data_path,
        keys: dict,
        bands=['vhvv', 'vh', 'vv'],
        transform=None
        ):
        
        self.path = data_path
        self.bands = bands
        
        with xr.open_dataset(os.path.join(self.path)) as ds:
            self.len = len(ds['point_id'])
        
        self.transform = transform
        self.keys = keys
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        series_data = []
        with xr.open_dataset(os.path.join(self.path)) as ds:
            for band in self.bands:    
                series_data.append(ds.isel(point_id=index)[band].to_numpy())
            label = ds.isel(point_id=index).label.item()
            
        encoded = np.zeros(len(self.keys.keys()))
        encoded[self.keys[label]] = 1
        signals = np.vstack((
            series_data
            ))
        
        return signals, encoded

    def convert_label(self, encoded_vec):
        label = list(self.keys.keys())[int(np.where(encoded_vec == 1)[0].item())]
        return label
    
    def get_ds(self):
        return xr.open_dataset(os.path.join(self.path))
