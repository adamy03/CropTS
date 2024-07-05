from momentfm import MOMENTPipeline
from momentfm.models.statistical_classifiers import fit_svm

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from data.dataset import CropTypeDataset
from models.moment_pipeline import *

import argparse
from argparse import Namespace
import random
import numpy as np
import os 
import pdb
import pickle

def control_randomness(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CropTypeTrainer:
    def __init__(self, args: Namespace):
        self.args = args
        
        #=== Load Data ===
        train_ds = CropTypeDataset(
            path=args.data_path,
            subset='train',
            bands=None,
            seq_len=args.seq_len,
            include_masks=self.args.masked
        )
        val_ds = CropTypeDataset(
            path=args.data_path,
            subset='val',
            bands=None,
            seq_len=args.seq_len,
            include_masks=self.args.masked
        )
        test_ds = CropTypeDataset(
            path=args.data_path,
            subset='test',
            bands=None,
            seq_len=args.seq_len,
            include_masks=self.args.masked
        )
        
        print('MASKED: ' + str(self.args.masked))
        
        self.train_dataloader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=1
        )
        self.val_dataloader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1
        )
        self.test_dataloader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=1
        )
        
        self.clf = None
        
        #=== Set model config ===
        
        self.model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                'task_name': 'classification',
                'n_channels': args.bands,
                'num_class': len(test_ds.get_keys()),
                'freeze_encoder': False if self.args.mode == 'full_finetuning' else True,
                'freeze_embedder': False if self.args.mode == 'full_finetuning' else True,
                'reduction': self.args.reduction,
                #Disable gradient checkpointing for finetuning or linear probing to 
                #avoid warning as MOMENT encoder is frozen
                'enable_gradient_checkpointing': False if self.args.mode in ['full_finetuning', 'linear_probing'] else True, 
            }
        )
        
        self.model.init()
        
        if self.args.from_checkpoint is not None:
            print('Loading checkpoint')
            state_dict = torch.load(self.args.from_checkpoint)
            state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        
        print('Model initialized, training mode: ', self.args.mode)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accelerator = None
        
        if self.args.mode == 'full_finetuning':
            print('Encoder and embedder are trainable')
            if self.args.lora:
                lora_config = LoraConfig(
                                        r=64,
                                        lora_alpha=32,
                                        target_modules=["q", "v"],
                                        lora_dropout=0.05,
                                        )
                self.model = get_peft_model(self.model, lora_config)
                print('LoRA enabled')
                self.model.print_trainable_parameters()

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.init_lr)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=self.args.max_lr, 
                total_steps=self.args.epochs*len(self.train_dataloader)
                )
            
            #set up model ready for accelerate finetuning
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader)
        
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.init_lr)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=self.args.max_lr, 
                total_steps=self.args.epochs*len(self.train_dataloader)
                )
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)
        self.log_file = open(os.path.join(self.args.output_path, f'{self.args.test_name}_{self.args.mode}_log.txt'), 'w')
        self.log_file.write(f'CropType training, mode: {self.args.mode}\n')
        self.log_file.write(f'Config: {str(self.args)}\n')

    def get_embeddings(self, dataloader: DataLoader):
        '''
        labels: [num_samples]
        embeddings: [num_samples x d_model]
        '''
        embeddings, labels = [], []
        self.model.to(self.device)

        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                if self.args.masked:
                    batch_x, batch_labels, batch_masks = batch
                    batch_masks = batch_masks.to(self.device)
                else: 
                    batch_x, batch_labels = batch
                    batch_masks = None 
                    
                # [batch_size x channels x 512]
                batch_x = batch_x.to(self.device).float()
                
                # [batch_size x num_patches x d_model (=1024)]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
                    output = self.model(batch_x, input_mask=batch_masks, reduction=self.args.reduction) 
                    
                # mean over patches dimension, [batch_size x d_model]
                embedding = output.embeddings.mean(dim=1)
                embeddings.append(embedding.detach().cpu().numpy())
                labels.append(batch_labels)        

        embeddings, labels = np.concatenate(embeddings), np.concatenate(labels)
        labels = np.argmax(labels, axis=1)
        labels = np.array([dataloader.dataset.get_keys(reversed=True)[label] for label in labels])
        
        return embeddings, labels
    
    def get_timeseries(self, dataloader: DataLoader, agg='mean'):
        '''
        mean: average over all channels, result in [1 x seq_len] for each time-series
        channel: concat all channels, result in [1 x seq_len * num_channels] for each time-series

        labels: [num_samples]
        ts: [num_samples x seq_len] or [num_samples x seq_len * num_channels]
        '''
        ts, labels = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                if self.args.masked:
                    batch_x, batch_labels, batch_masks = batch
                    batch_x = torch.stack([i * batch_masks for i in torch.unbind(batch_x, axis=1)], axis=1) # Apply masks across bands
                else: 
                    batch_x, batch_labels = batch
                    
                # [batch_size x channels x 512]
                if agg == 'mean':
                    batch_x = batch_x.mean(dim=1)
                    ts.append(batch_x.detach().cpu().numpy())
                elif agg == 'channel':
                    ts.append(batch_x.view(batch_x.size(0), -1).detach().cpu().numpy())
                
                labels.append(batch_labels.argmax(dim=1))        

        ts, labels = np.concatenate(ts), np.concatenate(labels)
        labels = np.array([dataloader.dataset.get_keys(reversed=True)[label] for label in labels])
        
        assert len(ts.shape) == 2 and len(labels.shape) == 1
        
        return ts, labels
    
    def train(self):
        for epoch in range(self.args.epochs):

            print(f'Epoch {epoch+1}/{self.args.epochs}')
            self.log_file.write(f'Epoch {epoch+1}/{self.args.epochs}\n')
            self.epoch = epoch + 1

            if self.args.mode == 'linear_probing':
                self.train_epoch_lp()
                self.evaluate_epoch()
            
            elif self.args.mode == 'full_finetuning':
                self.train_epoch_ft()
                self.evaluate_epoch()
                
            
            #break after training SVM, only need one 'epoch'
            elif self.args.mode == 'unsupervised_representation_learning':
                self.train_ul()
                break

            elif self.args.mode == 'svm':
                self.train_svm()
                break
            
            elif self.args.mode == 'random_forest':
                self.train_rf()
                break

            else:
                raise ValueError('Invalid mode, please choose svm, linear_probing, full_finetuning, or unsupervised_representation_learning')

#####################################training loops#############################################
    def train_epoch_lp(self):
        '''
        Train only classification head
        '''
        self.model.to(self.device)
        self.model.train()
        losses = []

        for batch in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
            self.optimizer.zero_grad()
            if self.args.masked:
                batch_x, batch_labels, batch_masks = batch
                batch_masks = batch_masks.to(self.device)
            else: 
                batch_x, batch_labels = batch
                batch_masks = None
    
            batch_x = batch_x.to(self.device).float()
            batch_labels = batch_labels.to(self.device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
                output = self.model(batch_x, input_mask=batch_masks, reduction=self.args.reduction)
                loss = self.criterion(output.logits, batch_labels)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        print('Train loss: ', avg_loss)
        self.log_file.write(f'Train loss: {avg_loss}\n')
        
    def train_epoch_ft(self):
        '''
        Train encoder and classification head (with accelerate enabled)
        '''
        self.model.to(self.device)
        self.model.train()
        losses = []

        for batch in tqdm(self.train_dataloader, total=len(self.train_dataloader),disable=not self.accelerator.is_local_main_process):
            self.optimizer.zero_grad()
            
            if self.args.masked:
                batch_x, batch_labels, batch_masks = batch
                batch_masks = batch_masks.to(self.device)
            else: 
                batch_x, batch_labels = batch
                batch_masks = None
            
            batch_x = batch_x.to(self.device).float()
            batch_labels = batch_labels.to(self.device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
                output = self.model(batch_x, input_mask=batch_masks, reduction=self.args.reduction)
                loss = self.criterion(output.logits, batch_labels)
                losses.append(loss.item())
            self.accelerator.backward(loss)
            
            self.optimizer.step()
            self.scheduler.step()

        avg_loss = np.mean(losses)
        print('Train loss: ', avg_loss)
        self.log_file.write(f'Train loss: {avg_loss}\n')
    
    def train_ul(self):
        '''
        Train SVM on top of MOMENT embeddings
        '''
        self.model.eval()
        self.model.to(self.device)

        #extract embeddings and label
        train_embeddings, train_labels = self.get_embeddings(self.train_dataloader)

        #fit statistical classifier
        self.clf = fit_svm(features=train_embeddings, y=train_labels)
        train_accuracy = self.clf.score(train_embeddings, train_labels)
        print('Train accuracy: ', train_accuracy)
        self.log_file.write(f'Train accuracy: {train_accuracy}\n')

    def train_svm(self):
        '''
        Train SVM on top of timeseries data
        '''
        train_embeddings, train_labels = self.get_timeseries(self.train_dataloader, agg=self.args.agg)
        self.clf = fit_svm(features=train_embeddings, y=train_labels)
        train_accuracy = self.clf.score(train_embeddings, train_labels)
        print('Train accuracy: ', train_accuracy)
        self.log_file.write(f'Train accuracy: {train_accuracy}\n')
    
    def train_rf(self):
        '''
        Train Random Forest on top of timeseries data
        
        '''
        train_embeddings, train_labels = self.get_timeseries(self.train_dataloader, agg=self.args.agg)
        
        self.clf = train_rf(data=train_embeddings, labels=train_labels)
        print(self.clf)
        train_accuracy = self.clf.score(train_embeddings, train_labels)
        print(True, 'Train accuracy: ', train_accuracy)
        self.log_file.write(f'Train accuracy: {train_accuracy}\n')
    
#####################################training loops#################################################

#####################################evaluate loops#################################################
    def test(self):
        if self.args.mode == 'unsupervised_representation_learning':
            test_embeddings, test_labels = self.get_embeddings(self.test_dataloader)
            test_accuracy = self.clf.score(test_embeddings, test_labels)
            print('Test accuracy: ', test_accuracy)
            self.log_file.write(f'Test accuracy: {test_accuracy}\n')

        elif self.args.mode == 'linear_probing' or self.args.mode == 'full_finetuning':
            self.evaluate_epoch(phase='test')

        elif self.args.mode =='svm':
            test_embeddings, test_labels = self.get_timeseries(self.test_dataloader, agg=self.args.agg)
            test_accuracy = self.clf.score(test_embeddings, test_labels)
            print('Test accuracy: ', test_accuracy)
            self.log_file.write(f'Test accuracy: {test_accuracy}\n')
        
        elif self.args.mode == 'random_forest':
            test_embeddings, test_labels = self.get_timeseries(self.test_dataloader, agg=self.args.agg)
            test_accuracy = self.clf.score(test_embeddings, test_labels)
            print('Test accuracy: ', test_accuracy)
            self.log_file.write(f'Test accuracy: {test_accuracy}\n')

        else:
            raise ValueError('Invalid mode, please choose linear_probing, full_finetuning, or unsupervised_representation_learning')
        
    def evaluate_epoch(self, phase='val', masked=False):
        if phase == 'val':
            dataloader = self.val_dataloader
        elif phase == 'test':
            dataloader = self.test_dataloader
        else:
            raise ValueError('Invalid phase, please choose val or test')

        self.model.eval()
        self.model.to(self.device)
        total_loss, total_correct = 0, 0

        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), disable=True if not self.accelerator and self.accelerator.is_local_main_process else False):

                if self.args.masked:
                    batch_x, batch_labels, batch_masks = batch
                    batch_masks = batch_masks.to(self.device)
                else: 
                    batch_x, batch_labels = batch
                    batch_masks = None
                    
                batch_x = batch_x.to(self.device).float()
                batch_labels = batch_labels.to(self.device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
                    output = self.model(batch_x, input_mask=batch_masks)
                    loss = self.criterion(output.logits, batch_labels)
                total_loss += loss.item()
                
                total_correct += (output.logits.argmax(dim=1) == batch_labels.argmax(dim=1)).sum().item()
                        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / len(dataloader.dataset)
        print(f'{phase} loss: {avg_loss}, {phase} accuracy: {accuracy}')
        self.log_file.write(f'{phase} loss: {avg_loss}, {phase} accuracy: {accuracy}\n')
    
    def run_inference(self, dataloader=None):
        if not dataloader:
            dataloader = self.test_dataloader
        
        self.model.eval()
        self.model.to(self.device)
        total_correct = 0
        predictions, labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), disable=True if not self.accelerator and self.accelerator.is_local_main_process else False):

                if self.args.masked:
                    batch_x, batch_labels, batch_masks = batch
                    batch_masks = batch_masks.to(self.device)
                else: 
                    batch_x, batch_labels = batch
                    batch_masks = None
                    
                batch_x = batch_x.to(self.device).float()
                batch_labels = batch_labels.to(self.device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32):
                    output = self.model(batch_x, input_mask=batch_masks)

                total_correct += (output.logits.argmax(dim=1) == batch_labels.argmax(dim=1)).sum().item()
                batch_preds = output.logits.argmax(dim=1)
                predictions.append(batch_preds.detach().cpu().numpy())
                labels.append(batch_labels.cpu().numpy())
        
        predictions, labels = np.concatenate(predictions), np.concatenate(labels)
        labels = np.array([dataloader.dataset.get_keys(reversed=True)[label] for label in labels])
        predictions = np.array([dataloader.dataset.get_keys(reversed=True)[pred] for pred in predictions])     
        accuracy = total_correct / len(dataloader.dataset)
        
        outputs = {
            'preds': predictions,
            'labels': labels,
            'accuracy': accuracy
        }
        
        # Save outputs to a pickle file
        with open(os.path.join(self.args.output_path, f'{self.args.test_name}_{self.args.mode}_test_outputs.pkl'), 'wb') as f:
            pickle.dump(outputs, f)   
            
        print(f'saved outputs at {self.args.output_path}/{self.args.test_name}_{self.args.mode}.pkl')
        
        
#####################################evaluate loops#################################################

    def save_checkpoint(self):
        if self.args.mode in ['svm', 'unsupervised_representation_learning']:
            raise ValueError('No checkpoint to save for SVM or unsupervised learning, as no training was done')
        
        path = self.args.output_path

        #mkdir if not exist
        if not os.path.exists(path):
            os.makedirs(path)

        #save parameter that requires grad 
        torch.save(self.model.state_dict(), os.path.join(path, f'{self.args.test_name}_{self.args.mode}_checkpoint.pth'))
        print('Model saved at ', path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--mode', type=str, default='full_finetuning', help='choose from linear_probing, full_finetuning, unsupervised_representation_learning, random_forest')
    parser.add_argument('--init_lr', type=float, default=1e-6)
    parser.add_argument('--max_lr', type=float, default=1e-4)
    parser.add_argument('--agg', type=str, default='channel', help='aggregation method for timeseries data for svm training, choose from mean or channel')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lora', action='store_true', help='enable LoRA')
    parser.add_argument('--reduction', type=str, default='concat', help='reduction method for MOMENT embeddings, choose from mean or max')
    #ptbxl dataset parameters
    parser.add_argument('--data_path', type=str, help='path to crop dataset')
    parser.add_argument('--cache_dir', type=str, help='path to cache directory to store preprocessed dataset')
    parser.add_argument('--output_path', type=str, help='path to save trained model and logs')
    parser.add_argument('--fs', type=int, default=100, help='sampling frequency, choose from 100 or 500')
    parser.add_argument('--code_of_interest', type=str, default='diagnostic_class')
    parser.add_argument('--output_type', type=str, default='single')
    parser.add_argument('--seq_len', type=int, default=512, help='sequence length for each sample, currently only support 512 for MOMENT')
    parser.add_argument('--bands', type=int, default=2, help='number of bands for input data')
    parser.add_argument('--load_cache', type=bool, default=True, help='whether to load cached dataset')
    parser.add_argument('--test_name', type=str, default='moment_classification', help='name of the test. will be used as checkpoint name')
    parser.add_argument('--from_checkpoint', type=str, default=None, help='path to model checkpoint')
    parser.add_argument('--masked', type=bool, default=False, help='train on masked')
    parser.add_argument('--run_mode', type=str, default='train', help='run_mode')
    
    args = parser.parse_args()
    control_randomness(args.seed)
    print('TEST_NAME:' + args.test_name +' MODE: ' + args.mode)
    trainer = CropTypeTrainer(args)
    
    if args.run_mode == 'train': 
        trainer.train()
        trainer.test()
        trainer.save_checkpoint()
    if args.run_mode == 'eval_results':
        trainer.run_inference()
        
    