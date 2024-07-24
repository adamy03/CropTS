import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Patch
from shapely import wkt
from src.models.model_utils import *
from src.data.ee_utils import *
from src.data.data_utils import *
from src.data.classes import LUCAS2018_LEVEL2_CODES, BANDS, LUCAS_10_CLASSES

# LUCAS2018_LEVEL2_CODES['NOT_CROP'] = 'Not Crop'

if __name__ == '__main__':
    config = {
    "task_name": "classification", 
    "model_name": "MOMENT", 
    "transformer_type": "encoder_only", 
    "d_model": None, 
    "seq_len": 512, 
    "patch_len": 8, 
    "patch_stride_len": 8, 
    "device": "cpu", 
    "transformer_backbone": "google/flan-t5-large", 
    "n_channels": 8,
    "num_class": 14,
    "freeze_encoder": False,
    "freeze_embedder": False,
    "reduction": 'concat',
    "enable_gradient_checkpointing": False
    }

    model_path = '/scratch/bbug/ayang1/experiments/lucas_fused_10/baseline_30_checkpoint/pytorch_model/mp_rank_00_model_states.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Namespace(**config)
    model = load_accelerator_model(config, model_path, device=device)
    model.to(device)
    
    test_ds = CropTypeDataset('/scratch/bbug/ayang1/datasets/lucas_fused_10/', subset='test')
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    model.eval()
    preds = []
    labels = []
    for i, batch in enumerate(tqdm(test_loader)):
        x = batch[0].to(device)
        with torch.no_grad():
            pred = model(x)
            labels.append(batch[1].argmax(axis=1).cpu())
            preds.append(pred.logits.argmax(axis=1).cpu())
            

    reverse = {v.argmax():k for k,v in LUCAS_10_CLASSES.items()}
    
    preds = torch.concat(preds, axis=0)
    preds = np.array([reverse[pred.item()] for pred in preds])
    
    labels = torch.concat(labels, axis=0)
    labels = np.array([reverse[label.item()] for label in labels])
    
    test_country = test_ds.country[:len(preds)]
    
    data = {
        'preds': preds,
        'labels': labels,
        'country': test_country
    }
    
    pd.DataFrame(data).to_csv('preds.csv', index=False)
    
    
    
    
    