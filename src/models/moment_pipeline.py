import sys
import os
import argparse
import torch
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from data.dataset import CropTypeDataset
from models.moment_pipeline import *

from argparse import Namespace

torch.manual_seed(0)

from models.classifier import *
from data.data_utils import *
from data.dataset import *
from tqdm import tqdm  
from momentfm import MOMENTPipeline
from momentfm.models.statistical_classifiers import fit_svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from argparse import Namespace

def control_randomness(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def load_moment(
    n_classes, 
    n_channels=3, 
    reduction='concat'
    ):
    
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={
            'task_name': 'classification',
            'n_channels': n_channels,
            'num_class': n_classes,
            'freeze_encoder': True,
            'freeze_embedder': True,
            'reduction': reduction
        },
    )
    model.init()
    
    return model

def run_moment(
    model, 
    device,
    dataloader,
    reduction='mean'
    ):
    embeddings, logits, labels = [], [], []
    
    model.eval()
    with torch.no_grad():
        for batch_x, batch_labels in tqdm(dataloader, total=len(dataloader)):
            batch_x = batch_x.to(device).float()

            output = model(batch_x) # [batch_size x d_model (=1024)]
            embedding = output.embeddings
            logit = output.logits
            
            logits.append(logit.detach().cpu().numpy())
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(batch_labels)        

    embeddings, logits, labels = np.concatenate(embeddings), np.concatenate(logits), np.concatenate(labels)
    
    if reduction == 'mean':
        embeddings = np.mean(embeddings, axis=1)
        
    return embeddings, logits, labels

def moment_pipeline(**kwargs):

    data_name = kwargs['subset'] + '_signals.npy'
    label_name = kwargs['subset'] + '_labels.npy'
    
    print('Loading data...')
    data = load_crop_data(
        data_path=kwargs['data_path'],
        seq_len=kwargs['seq_len'],
        bands=kwargs['bands'],
        data_name=data_name,
        label_name=label_name
        )
    
    if not 'n_classes' in kwargs:
        n_classes = len(data.get_keys())
            
    print('Loading MOMENT...')
    moment_model = load_moment(
        n_classes=n_classes, 
        n_channels=kwargs['n_channels'],
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = DataLoader(data, batch_size=kwargs['batch_size'], shuffle=False, num_workers=1)
    moment_model.to(device)
    
    print('Running MOMENT...')
    embeddings, logits, labels = run_moment(
        model=moment_model, 
        device=device,
        dataloader=data_loader
    )
    
    moment_outputs = {
        'embeddings': embeddings,
        'logits': logits,
        'labels': labels,
        'keys': data.get_keys()
    }
    
    return moment_outputs

def train_linear(
    model,
    loss_fn,
    device,
    optim,
    epochs,
    train_loader,
    test_loader,
    tune_mode=False
    ):
    
    model = model.to(device)
    train_losses = []
    test_losses = []
    
    for epoch in tqdm(range(epochs)):
        train_epoch_loss = []
        test_epoch_loss = []
        model.train()
        
        for idx, batch in enumerate(train_loader):
            optim.zero_grad()
            embedding, label = batch
            embedding, label = embedding.to(device), label.to(device)
            output = model(embedding)
            loss = loss_fn(output, label)
            loss.backward()
            optim.step()
            # train_losses.append(loss.item())
            train_epoch_loss.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                embedding, label = batch
                embedding, label = embedding.to(device), label.to(device)
                output = model(embedding)
                loss = loss_fn(output, label)
                # test_losses.append(loss.item())
                test_epoch_loss.append(loss.item())

        epoch_train_loss = np.mean(train_epoch_loss)
        epoch_test_loss = np.mean(test_epoch_loss)  
        train_losses.append(epoch_train_loss) 
        test_losses.append(epoch_test_loss)
        
        print(f'epoch: {epoch+1}, train loss: {epoch_train_loss:.5f}, test_loss:{epoch_test_loss:.5f}')
    
    return model, optim, train_losses, test_losses

def train_svm(
    embeddings, 
    labels
):
    best_model = fit_svm(embeddings, labels)
    
    return best_model

def train_rf(data, labels):
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=150
    )
    grid_search = GridSearchCV(
        rf,
        {
            'n_estimators': [100, 150, 200],
            'max_depth': [50, 100, 150, 200],
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

def load_embedding_ds(
    path,
    from_results=True
):
    if from_results:
        results = torch.load(path, map_location=torch.device('cpu'))    
        train_data = results['train_outputs']['embeddings']
        train_labels = results['train_outputs']['labels']
        test_data = results['test_outputs']['embeddings']
        test_labels = results['test_outputs']['labels'] 
        keys = results['train_outputs']['keys']
        
        train_ds = EmbeddingDataset(train_data, train_labels)
        test_ds = EmbeddingDataset(test_data, test_labels)
        
        return train_ds, test_ds, keys

    else:
        train_ds = EmbeddingDataset(
            embeddings=np.load(os.path.join(path, 'train_embed.npy')), 
            labels=np.load(os.path.join(path, 'train_labels.npy'))
        )
    
        test_ds = EmbeddingDataset(
            embeddings=np.load(os.path.join(path, 'test_embed.npy')), 
            labels=np.load(os.path.join(path, 'test_labels.npy'))
        )
        
        return train_ds, test_ds
