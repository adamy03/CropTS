import sys
from networkx import parse_graphml
import torch
import numpy as np
import pickle as pkl
import argparse

from torch.utils.data import DataLoader
from data.dataset import *
from data.data_utils import *
from operator import length_hint
from momentfm import MOMENTPipeline
from tqdm import tqdm

torch.manual_seed(0)

BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Parser
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('--data_path', help='training file', required=True)
    parser.add_argument('--out_path', default='./', help='output path', required=True)
    args = vars(parser.parse_args())
    
    # load model
    model = load_model()
    
    # generate dataset
    with xr.open_dataset(args['data_path']) as ds:
        labels = np.unique(ds.label)
        keys = encode_labels(labels)
    
    ds = CropTypeDataset(args['data_path'], keys=keys)
    data_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # run model
    model.to(DEVICE).float()

    print('Running inference...')
    train_embeddings, train_logits, train_labels = get_outputs(model, data_loader)

    # export outputs
    model_outputs = {
        'embeddings': train_embeddings,
        'logits': train_logits,
        'labels': train_labels
    }
        
    with open(os.path.join(args['out_path'], 'outputs.pkl'), 'wb') as f:
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

def load_model():
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={
            'task_name': 'classification',
            'n_channels': 3,
            'num_class': 6,
            'freeze_encoder': True,
            'freeze_embedder': True,
            'reduction': 'concat'
        },
    )

    print('Loading model...')
    model.init()
    
    return model

if __name__=='__main__':
    main()