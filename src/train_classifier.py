import numpy as np
import argparse
from models.model_utils import *

class CropClassifier:
    def __init__(self, args):
        self.args = args
        self.train_data, self.test_data, self.train_labels, self.test_labels = self.load_data()
        self.train_data = self.train_data[:, :, :self.args.seq_len].mean(axis=1)
        self.test_data = self.test_data[:, :, :self.args.seq_len].mean(axis=1)
        self.clf = None
        self.accuracy = None
        
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)
            
        self.log_file = open(
            os.path.join(self.args.output_path, f"{self.args.test_name}_log.txt"),
            "w",
        )
        self.log_file.write(f"CropType training, mode: {self.args.mode}\n")
        
    def load_data(self):
        train_data = np.load(os.path.join(self.args.data_path, 'train_signals.npy'), allow_pickle=True)
        test_data = np.load(os.path.join(self.args.data_path, 'test_signals.npy'), allow_pickle=True)
        train_labels = np.load(os.path.join(self.args.data_path, 'train_labels.npy'), allow_pickle=True)
        test_labels = np.load(os.path.join(self.args.data_path, 'test_labels.npy'),
        allow_pickle=True)
        
        return train_data, test_data, train_labels, test_labels
    
    def train(self):
        print("Training classifier...")
        if self.args.mode == 'random_forest':
            self.clf = train_rf(self.train_data, self.train_labels)
            print("Model Params: ", self.clf.get_params())
            self.log_file.write(f"Config: {str(self.clf.get_params())}\n")
        else:
            raise ValueError(f'Invalid mode: {self.args.mode}')
    
    def evaluate(self):
        if self.clf is None:
            raise ValueError('Run train beofre evaluating')
        
        print("Evaluating classifier...")
        if self.args.mode == 'random_forest':
            train_accuracy = self.clf.score(self.train_data, self.train_labels)
            test_accuracy = self.clf.score(self.test_data, self.test_labels)
            print("Train Accuracy: ", train_accuracy)
            print("Test Accuracy: ", test_accuracy)
            self.log_file.write(f"Accuracy: {test_accuracy}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data", help="Path to data")
    parser.add_argument("--output_path", type=str, default="output", help="Path to output")
    parser.add_argument("--mode", type=str, default="random_forest", help="Mode to run")
    parser.add_argument("--test_name", type=str, default="test", help="Name of the test")
    parser.add_argument("--seq_len", type=int, default=36, help="Length of input sequence")
    args = parser.parse_args()
    
    classifier = CropClassifier(args)
    classifier.train()
    classifier.evaluate()