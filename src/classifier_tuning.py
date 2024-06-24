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

torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Train network.')
parser.add_argument('--data_path', help='training file', required=True)
parser.add_argument('--num_samples', default=10, type=int, required=False)
parser.add_argument('--max_epochs', default=25, type=int, required=False)
parser.add_argument('--gpus', default=1, type=int, required=False)
parser.add_argument('--input_len', default=2048, type=int, required=False)

def load_data(data_path):
    train_ds = EmbeddingDataset(
        embeddings=np.load(os.path.join(data_path, 'train_embed.npy')), 
        labels=np.load(os.path.join(data_path, 'train_labels.npy'))
        )
    
    test_ds = EmbeddingDataset(
        embeddings=np.load(os.path.join(data_path, 'test_embed.npy')), 
        labels=np.load(os.path.join(data_path, 'test_labels.npy'))
        )
    
    return train_ds, test_ds

def train_classifier(config):
    data_path = config['data_path']
    batch_size = config['batch_size']
    input_dim = config['input_dim']
    batch_size = config['batch_size']
    hidden_dim = config['hidden_dim']
    lr = config['lr']
    epochs = config['epochs']
    
    # Load data
    train_ds, test_ds = load_data(data_path)
    train_subset, val_subset = random_split(
        train_ds, [int(len(train_ds) * 0.8), len(train_ds) - int(len(train_ds) * 0.8)]
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=True, num_workers=1
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
    
    # Training loop
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
     
    for epoch in range(epochs):
        train_e_loss = []
        train_e_steps = 0
        train_e_acc = 0
        
        val_e_loss = []
        val_e_steps = 0
        val_e_acc = 0
        
        model.train()
        
        # Training
        for idx, batch in enumerate(train_loader):
            # Batch data
            embeddings, labels = batch
            embeddings, labels = embeddings.to(device),  labels.to(device)
            
            # Forward/backward pass
            optim.zero_grad()
            outputs = model(embeddings)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()
            
            # Log loss
            train_e_loss.append(loss.item())
            train_e_steps += 1
            train_e_acc += (outputs.argmax(1) == labels.argmax(1)).sum().item()
                        
        # Validation
        model.eval()
        for idx, batch in enumerate(val_loader):
            embeddings, labels = batch
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            loss = loss_fn(outputs, labels)
            
            val_e_loss.append(loss.item())
            val_e_steps += 1
            val_e_acc += (outputs.argmax(1) == labels.argmax(1)).sum().item()
            
        # Raytune report
        checkpoint_data = {
            'epoch': epoch,
            'train_loss': train_loss.append(np.mean(train_e_loss)),
            'train_acc': train_e_acc / train_e_steps,
            'val_loss': np.mean(val_e_loss),
            'val_acc': val_e_acc / val_e_steps,
            'test_acc': test_accuracy(model, data_path, device)
        }
        
        train.report(checkpoint_data)
        

def test_accuracy(model, data_path, device='cpu'):
    trainset, testset = load_data(data_path)
    test_loader = DataLoader(
        testset, batch_size=15, shuffle=False, num_workers=1
    )
    
    acc = 0
    steps = 0

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            embeddings, labels = batch
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            acc += (outputs.argmax(1) == labels.argmax(1)).sum().item()
            steps += 1

    return float(acc / steps)

def main():
    # Parser
    args = parser.parse_args()
    
    # Model config
    config = {
        'data_path': args.data_path,
        'input_dim': args.input_len,
        'batch_size': tune.choice([8, 16, 32, 64]),
        'hidden_dim': tune.choice([512, 1024]),
        'lr': tune.loguniform(1e-4, 1e-1),
        'epochs': args.max_epochs,        
    }

    ray.init(configure_logging=False)

    tuner = tune.Tuner(
        train_classifier, 
        tune_config=tune.TuneConfig(
            metric='test_acc',
            mode='max',
            num_samples=args.num_samples,
        ),
        param_space=config
        )
    
    results = tuner.fit()
    
    print("Best hyperparameters found were: ", results.get_best_result().config)
    
if __name__ == "__main__":
    main()