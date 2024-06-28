import os
import torch
import torch.nn as nn
import argparse
import ray
from torch.utils.data import random_split, DataLoader
import numpy as np
from data.dataset import EmbeddingDataset
from models.classifier import CropTypeClassifier
from ray import tune, train
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search import ConcurrencyLimiter
from models.moment_pipeline import *

torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Train network.')
parser.add_argument('--data_path', help='training file', required=True)
parser.add_argument('--num_samples', default=10, type=int, required=False)
parser.add_argument('--epochs', default=25, type=int, required=False)
parser.add_argument('--gpus', default=1, type=int, required=False)
parser.add_argument('--input_dim', default=2048, type=int, required=False)

def ray_train_classifier(config):
    data_path = config['data_path']
    batch_size = config['batch_size']
    input_dim = config['input_dim']
    hidden_dim = config['hidden_dim']
    lr = config['lr']
    epochs = config['epochs']
    
    # Load data
    train_ds, test_ds, keys = load_embedding_ds(data_path, from_results=True)
    
    train_subset, val_subset = random_split(
        train_ds, [int(len(train_ds) * 0.8), len(train_ds) - int(len(train_ds) * 0.8)]
    )

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=True, num_workers=1
    )
    
    # Hyper parameters
    model = CropTypeClassifier(
        input_dim=input_dim,
        n_classes=train_ds.labels.shape[1],
        hidden_dim=hidden_dim,
    )
    
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
        
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)

    model = model.to(device)
     
    for epoch in range(epochs):
        train_e_loss = []
        val_e_loss = []
        val_e_acc = 0
        val_e_steps = 0
        
        model.train()
        
        # Training
        for idx, batch in enumerate(train_loader):
            # Batch data
            embeddings, labels = batch
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            # Forward/backward pass
            optim.zero_grad()
            outputs = model(embeddings)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()
            
            # Log loss
            train_e_loss.append(loss.item())
                        
        # Validation
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                embeddings, labels = batch
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                loss = loss_fn(outputs, labels)
                
                val_e_loss.append(loss.item())
                val_e_acc += (outputs.argmax(1) == labels.argmax(1)).sum().item()
                val_e_steps += labels.shape[0]
                
        # Raytune report
        checkpoint_data = {
            'epoch': epoch,
            'train_loss': np.mean(train_e_loss),
            'val_loss': np.mean(val_e_loss),
            'val_acc': val_e_acc / val_e_steps,
            'test_acc': test_linear_accuracy(model, test_loader, device)
        }
        
        train.report(checkpoint_data)

def run_tuner(
    tuner_config
):
    data_path = tuner_config['data_path']
    input_dim = tuner_config['input_dim']
    epochs = tuner_config['epochs']
    num_samples = tuner_config['num_samples']
    
    # Model config
    config = {
        'data_path': data_path,
        'input_dim': input_dim,
        'batch_size': tune.choice([8, 16, 32, 64]),
        'hidden_dim': tune.choice([512, 1024, 2048]),
        'lr': tune.loguniform(1e-5, 1e-3),
        'epochs': epochs,        
    }

    ray.init(configure_logging=False)

    tuner = tune.Tuner(
        ray_train_classifier, 
        tune_config=tune.TuneConfig(
            metric='test_acc',
            mode='max',
            num_samples=num_samples,
        ),
        param_space=config
        )
    
    results = tuner.fit()
    
    return results.get_best_result()

def main():
    args = vars(parser.parse_args())
    print(run_tuner(args))
    
if __name__ == "__main__":
    main()