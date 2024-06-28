import torch
import numpy as np
import pickle as pkl
import argparse
import sys

from torch.utils.data import DataLoader
from torchvision.transforms import v2
from data.dataset import *
from data.data_utils import *
from momentfm import MOMENTPipeline
from tqdm import tqdm

torch.manual_seed(0)

BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Train network.')
parser.add_argument('--data_path', help='training file', required=True)
parser.add_argument('--out_dir', default='./', help='output path', required=True)
parser.add_argument('--out_name', default='outputs', help='output filename', required=False)
parser.add_argument('--model_path', default=None, help='model path', required=False)
parser.add_argument('--n_channels', default=3, type=int, help='number of channels', required=False)
parser.add_argument('--n_classes', default=None, type=int, help='number of classes', required=False)

def main():
    # Parser
    args = parser.parse_args()
    n_classes = args.n_classes
    
    # Load data
    data = np.load(os.path.join(args.data_path, 'train_signals.npy'))
    labels = np.load(os.path.join(args.data_path, 'train_labels.npy'))
    ds = CropTypeDataset(data, labels, seq_len=16)
    data_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    if n_classes is None:
        n_classes = len(ds.get_keys())

    # load model
    model = load_moment(
        n_classes=n_classes,
        n_channels=args.n_channels
    )
    
    # print(model)
    
    # run model
    model.to(DEVICE)

    print('Running inference...')
    train_embeddings, train_logits, train_labels = get_outputs(model, data_loader)

    # export outputs
    model_outputs = {
        'embeddings': train_embeddings,
        'logits': train_logits,
        'labels': train_labels,
        'keys': ds.get_keys()
    }
        
    with open(os.path.join(args.out_dir, args.out_name+'.pkl'), 'wb') as f:
        pkl.dump(model_outputs, f)
        
    return 

def get_outputs(model, dataloader):
    embeddings, logits, labels = [], [], []
    
    model.eval()
    with torch.no_grad():
        for batch_x, batch_labels in tqdm(dataloader, total=len(dataloader)):
            batch_x = batch_x.to(DEVICE).float()

            output = model(batch_x) # [batch_size x d_model (=1024)]
            embedding = output.embeddings
            logit = output.logits
            
            logits.append(logit.detach().cpu().numpy())
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(batch_labels)        

    embeddings, logits, labels = np.concatenate(embeddings), np.concatenate(logits), np.concatenate(labels)
    return embeddings, logits, labels

def load_moment(n_classes, n_channels=3, reduction='concat'):
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

    print('Loading model...')
    model.init()
    
    return model

if __name__=='__main__':
    main()