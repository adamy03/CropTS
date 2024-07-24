import numpy as np
import pandas as pd
import h5py
import os
import sys
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from .classes import BANDS, LUCAS_10_CLASSES

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
        self.include_masks = include_masks
        
        self.bands = bands
        if not bands:
            self.bands = BANDS
        
        if not subset in ['train', 'val', 'test']:
            raise ValueError('subset must be "train", "val", or "test"')

        # Dataset
        self.dataset = pd.read_csv(os.path.join(path, f'{subset}.csv'))
        
        # Masks
        if include_masks:
            self.masks = np.load(os.path.join(path, f'{subset}_masks.npy'), allow_pickle=True)
        
        # Labels
        self.labels = self.dataset['LABEL'].values
        self.no_unique_labels = len(np.unique(self.labels))
        self.keys = LUCAS_10_CLASSES
        self.labels = np.array([self.keys[label] for label in self.labels])
        
        # Band data
        band_data = []
        for band in self.bands:
            band_df = self.dataset.loc[:, self.dataset.columns.str.contains(band)]
            
            if len(band_df.columns) == 0:
                raise ValueError(f'Band {band} not found in dataset')
            
            cols = list(band_df.columns)
            cols.sort(key=lambda x: int(x.split('_')[0]))
            band_df = band_df.reindex(cols, axis=1)
            band_data.append(band_df.to_numpy()[:, np.newaxis, :])
            
        self.data = np.concatenate(band_data, axis=1)
        self.country = self.dataset['country'].values
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        data = self.data[index]
        
        if self.seq_len:
            data = data[:, :self.seq_len]
        
        data = torch.from_numpy(data.astype(np.float32))
        label = torch.from_numpy(self.labels[index])
        
        if self.transform:
            data = self.transform(data)
            label = self.transform(label)
        
        if self.include_masks: 
            mask = torch.from_numpy(self.masks[index])
            
            if self.seq_len:
                mask = mask[:, :self.seq_len]

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
    