import signal
import pandas as pd
import numpy as np
import pickle as pkl
import os
import h5py

from typing import Iterable
from tqdm import tqdm

def add_lucas_labels(
    signals: str, 
    labels: str,
):
    """Adds LUCAS labels to SAR signals based on POINT_ID

    Args:
        signals (str): df of SAR signals
        labels (str): LUCAS labels 

    Returns:
        _type_: dataframe with LUCAS labels 
    """
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
        
def generate_s1_dataset(
    data: pd.DataFrame,
    output_path: str,
    output_name: str,
    columns = ['vhvv', 'vh', 'vv', 'label'],
    indicies = [(1, 37), (37, 73), (73, 109), 'LABEL'],
    ts_length = 36,
    additional_columms = None,
):
    """Creates hdf for moment dataset

    Args:
        data (pd.DataFrame): Crop type labeled sar signals
        output_path (str): Folder to save hdf
        output_name (str): Name of hdf file
    """    

    with h5py.File(os.path.join(output_path, output_name+'.h5'), 'w') as f:
    # Create a dataset in the file
        length = len(data)
        for column, index in zip(columns, indicies):
            if column == 'label':
                f.create_dataset(
                    column, 
                    (length), 
                    data=data[index].to_numpy().astype(h5py.string_dtype())
                )
            else:
                f.create_dataset(
                    column, 
                    (length, ts_length), 
                    data=data.loc[:, data.columns[index[0]:index[1]]].to_numpy()
                )
    
def encode_labels(labels):
    """Convert labels to one hot encoded vectors

    Args:
        labels (_type_): list of labels to convert

    Returns:
        _type_: Dict mapping labels to one hot encoded indicies
    """
    # Return dict mapping labels to one hot encoded indicies
    labels = np.unique(labels)
    encoded_vecs = np.identity(len(labels))
    keys = dict(zip(labels, [encoded_vecs[i, :] for i in range(len(labels))]))
    
    return keys

def convert_label(encoded_vec, keys):
    """Convert one hot encoded vector to associated label

    Args:
        encoded_vec (_type_): Vector to convert
        keys (_type_): Mapping of labels to one hot encoded indicies

    Returns:
        _type_: Label
    """
    label = list(keys.keys())[int(np.where(encoded_vec == 1)[0].item())]
    return label

def load_data(
    data_path: str,
):
    """Load hdf data to memory 

    Args:
        data_path (str): hdf file path

    Returns:
        _type_: Dict of data
    """
    with h5py.File(data_path, 'r') as f:
        keys = [key for key in f.keys()]
        data = []
        for key in keys:
            data.append(f[key][:])
            
    return dict(zip(keys, data)) 
