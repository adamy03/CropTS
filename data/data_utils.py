import signal
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle as pkl
import os
import h5py
import xarray as xr

from typing import Iterable
from tqdm import tqdm

def add_lucas_labels(
    signals: str, 
    labels: str,
):
    print('Loading files')
    
    print(f'Creating dataset of size {len(signals)}')
    
    matched_labels = []
    pbar = tqdm(total=len(signals))

    for index, row in signals.iterrows():
        label = str(labels.loc[labels['POINT_ID']==row['POINT_ID']]['LC1'].item())
        if 'B' not in label:
            label = 'NOT_CROP'
            
        matched_labels.append(label)
        
        pbar.update(1)
    
    signals['LABEL'] = matched_labels
    
    return signals
        
def generate_lucas_s1_dataset(
    data: pd.DataFrame,
    output_path: str,
):
    assert set(['POINT_ID', 'LABEL']) <= set(data.columns) # Check for labels 
    
    vhvv = data.loc[:, data.columns[1:37]].to_numpy()
    vh = data.loc[:, data.columns[37:73]].to_numpy()
    vv = data.loc[:, data.columns[73:109]].to_numpy()
    point_ids = data['POINT_ID'].to_numpy()
    labels = data['LABEL'].to_numpy()
    dates = np.array([x.split('_')[1] for x in data.columns[1:37]])
    
    with h5py.File(output_path, 'w') as f:
    # Create a dataset in the file
        length = len(labels)
        f.create_dataset('vhvv', (length, 36), data=vhvv)
        f.create_dataset('vh', (length, 36), data=vh)
        f.create_dataset('vv', (length, 36), data=vv)
        f.create_dataset('point_id', (length), data=point_ids)
        f.create_dataset('label', (length), data=labels)
        f.create_dataset('date', (36), data=dates)
    
def encode_labels(labels):
    # Return dict mapping labels to one hot encoded indicies
    labels = np.unique(labels)
    keys = dict(zip(labels, np.arange(0, len(labels))))
    
    return keys

def convert_label(encoded_vec, keys):
    label = list(keys.keys())[int(np.where(encoded_vec == 1)[0].item())]
    return label

def load_data(
    data_path: str,
):
    with h5py.File(data_path, 'r') as f:
        keys = [key for key in f.keys()]
        data = []
        for key in keys:
            data.append(f[key][:])
            
    return dict(zip(keys, data)) 
