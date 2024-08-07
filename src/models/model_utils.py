import os
from matplotlib import figure
import torch
import numpy as np
import torch
import random
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('../')
torch.manual_seed(0)
sns.set_style()

from sklearn.metrics import confusion_matrix
from data.data_utils import *
from data.dataset import *
from tqdm import tqdm  
from momentfm import MOMENT, MOMENTPipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from argparse import Namespace
from accelerate import Accelerator

def control_randomness(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def train_rf(data, labels):
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=150
    )
    grid_search = GridSearchCV(
        rf,
        {
            'n_estimators': [100, 150, 200, 250, 300],
            'max_depth': [100, 150, 200, 250, 300],
        },
        cv=5,
        n_jobs=10,
    )
    
    grid_search.fit(data, labels)
    
    return grid_search.best_estimator_

def test_linear_accuracy(model, test_loader, device='cpu', show_bar=False):
    acc = 0
    steps = 0
    
    if show_bar:
        pbar = tqdm(total=len(test_loader))
    
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            embeddings, labels = batch
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            acc += (outputs.argmax(1) == labels.argmax(1)).sum().item()
            steps += labels.shape[0]
            
            if show_bar:
                pbar.update(1)

    return float(acc / steps)

def load_accelerator_model(config, accelerator_model_path, device='cpu'):
    """
    Sample config:
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
        "n_channels": 7,
        "num_class": 12,
        "freeze_encoder": False,
        "freeze_embedder": False,
        "reduction": 'concat',
        "enable_gradient_checkpointing": False
    }

    config = Namespace(**config)
    """
    model = MOMENT(config)
    with open(accelerator_model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=torch.device(device))['module'])
    
    return model  
    
def confusion_plot(pred, true, labels, title):
    cf_matrix = confusion_matrix(
        true,
        pred
        )
    
    fig = plt.figure(figsize=(9,7))
    
    sns.heatmap(
        cf_matrix, 
        annot=True, fmt='d',
        xticklabels=labels,
        yticklabels=labels
    )
    
    plt.title(title)
    
    return fig
    
    