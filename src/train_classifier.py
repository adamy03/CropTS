import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import torch

torch.manual_seed(0)

from models.classifier import *
from torch.utils.data import DataLoader
from data.data_utils import *
from data.dataset import *
from tqdm import tqdm

sns.set_theme()
cmap = sns.color_palette("tab10", as_cmap=True)

parser = argparse.ArgumentParser(description='Classify embeddings')
parser.add_argument('--data_path', help='path to embeddings', required=True)
parser.add_argument('--save_path', default=None, help='path to save model', required=False)
parser.add_argument('--test_name', default=None, help='name of test', required=False)
parser.add_argument('--input_len', default=None, type=int, help='path to load model', required=False)
parser.add_argument('--batch_size', default=32, type=int, help='batch size', required=False)
parser.add_argument('--hidden_dim', default=512, type=int, help='hidden dimension', required=False)
parser.add_argument('--epochs', default=100, type=int, help='number of epochs', required=False)
parser.add_argument('--lr', default=0.001, type=float, help='learning rate', required=False)

def main():
    args = parser.parse_args()
    
    # Load Data
    print('Loading data...')
    train_embeddings = np.load(os.path.join(args.data_path, 'train_embed.npy'))
    train_labels = np.load(os.path.join(args.data_path, 'train_labels.npy'))
    test_embeddings = np.load(os.path.join(args.data_path, 'test_embed.npy'))
    test_labels = np.load(os.path.join(args.data_path, 'test_labels.npy'))
    
    train_ds = EmbeddingDataset(embeddings=train_embeddings, labels=train_labels)
    test_ds = EmbeddingDataset(embeddings=test_embeddings, labels=test_labels)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    # Model parameters
    print('Loading model...')
    model = CropTypeClassifier(
        input_dim=args.input_len,
        n_classes=train_labels.shape[1],
        hidden_dim=512,
    )
    
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = args.lr
    epochs = args.epochs
    optim = torch.optim.Adam(model.parameters(), lr=lr) 

    # Train Model
    print('Training model...')
    model, optimizer, train_losses, test_losses = train_classifier_head(
        model,
        loss_fn,
        device,
        optim,
        epochs,
        train_loader,
        test_loader,
    )
    
    if args.save_path:
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses
        }
        torch.save(state, os.path.join(args.save_path, args.test_name + '.pth'))
        
def train_classifier_head(
    model,
    loss_fn,
    device,
    optim,
    epochs,
    train_loader,
    test_loader,
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
            train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                embedding, label = batch
                embedding, label = embedding.to(device), label.to(device)
                output = model(embedding)
                loss = loss_fn(output, label)
                test_losses.append(loss.item())

        print(f'epoch: {epoch+1}, train loss: {np.mean(train_losses):.5f}, test_loss:{np.mean(test_losses):.5f}')
    
    return model, optim, train_losses, test_losses
        
if __name__ == '__main__':
    main()