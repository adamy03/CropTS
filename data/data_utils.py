import signal
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle as pkl
import os
import xarray as xr

from typing import Iterable
from tqdm import tqdm

def add_labels(
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
        
def generate_datasets(
    data: pd.DataFrame,
):
    assert set(['POINT_ID', 'LABEL']) <= set(data.columns) # Check for labels 
    
    vhvv = data.loc[:, data.columns[1:37]].to_numpy()
    vh = data.loc[:, data.columns[37:73]].to_numpy()
    vv = data.loc[:, data.columns[73:109]].to_numpy()
    point_ids = data['POINT_ID'].to_numpy()
    labels = data['LABEL'].to_numpy()
    dates = np.array([x.split('_')[1] for x in data.columns[1:37]])
    
    ds = xr.Dataset(
        data_vars=dict(
            vhvv=(['point_id', 'dates'], vhvv),
            vv=(['point_id', 'dates'], vv),
            vh=(['point_id', 'dates'], vh),
            label=(['point_id'], labels)
        ),
        coords=dict(
            point_id=point_ids,
            dates=dates
        ),
        attrs=dict(description='SAR Bands'),
    )
    
    return ds