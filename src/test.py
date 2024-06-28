import torch
import argparse

from models.classifier import *
from models.moment_pipeline import *

parser = argparse.ArgumentParser(description='Train linear classifier')
parser.add_argument('--data_path', help='path to data', required=True)
parser.add_argument('--seq_len', default=32, type=int, help='sequence length', required=False)
parser.add_argument('--bands', default=2, type=int, help='number of bands', required=False)
parser.add_argument('--hidden_dim', default=512, type=int, help='hidden dimension', required=False)
parser.add_argument('--lr', default=0.001, type=float, help='learning rate', required=False)
parser.add_argument('--epochs', default=100, type=int, help='number of epochs', required=False)
parser.add_argument('--batch_size', default=32, type=int, help='batch size', required=False)
parser.add_argument('--save_path', default='.', help='path to save results', required=False)
parser.add_argument('--test_name', default='test', help='name of test', required=False)

def train_linear_classifier(
    config
):
    seq_len = config['seq_len']
    bands = list(range(config['bands']))
    data_path = config['data_path'] 
    n_channels = len(bands)
    hidden_dim = config['hidden_dim']
    lr = config['lr']
    epochs = config['epochs']
    batch_size = config['batch_size']
    save_path = config['save_path']
    test_name = config['test_name']
    
    train_outputs = moment_pipeline(
        data_path=data_path,
        seq_len=seq_len,
        bands=bands,
        subset='train',
        n_channels=n_channels,
        batch_size=128
    )
    
    test_outputs = moment_pipeline(
        data_path=data_path,
        seq_len=seq_len,
        bands=bands,
        subset='test',
        n_channels=n_channels,
        batch_size=128
    )
    
    model = CropTypeClassifier(
        n_classes=len(train_outputs['keys']),
        input_dim=train_outputs['embeddings'].shape[1],
        hidden_dim=hidden_dim
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = lr
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_loader = DataLoader(
        EmbeddingDataset(train_outputs['embeddings'], train_outputs['labels']),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )
    
    test_loader = DataLoader(
        EmbeddingDataset(test_outputs['embeddings'], test_outputs['labels']),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )
    
    model, optim, train_losses, test_losses = train_linear(
        model,
        loss_fn,
        device,
        optim=optim,
        epochs=epochs,
        train_loader=train_loader,
        test_loader=test_loader,
    )
    
    accuracy = test_linear_accuracy(
        model, 
        test_loader=test_loader, 
        device=device
        ) 
    
    classifier_results = {
        'model': model.state_dict(),
        'optimizer': optim.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'accuracy': accuracy
        }
    
    results = {
        'train_outputs': train_outputs,
        'test_outputs': test_outputs,
        'classifier_results': classifier_results,
        'config': config
    }
    
    torch.save(results, os.path.join(save_path, test_name+'.pth'))
    
if __name__=='__main__':
    args = vars(parser.parse_args())
    train_linear_classifier(args)
    