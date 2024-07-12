import torch
import argparse
import random
import numpy as np
import os
import pdb
import pickle
import wandb

from momentfm import MOMENTPipeline
from data.dataset import CropTypeDataset
from models.model_utils import *
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from argparse import Namespace

WANDB_PROJECT = "moment-crop"

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

        ### Load Data ###
        train_ds, val_ds, test_ds = self.load_datasets()

        self.train_dataloader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=1
        )
        self.val_dataloader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1
        )
        self.test_dataloader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=1
        )

        ### Model Setup ###
        self.model = self.initialize_model()
        print(self.model)

        if self.args.mode == "full_finetuning":
            print("Encoder and embedder are trainable")
            if self.args.lora:
                lora_config = LoraConfig(
                    r=64,
                    lora_alpha=32,
                    target_modules=["q", "v"],
                    lora_dropout=0.05,
                )
                self.model = get_peft_model(self.model, lora_config)
                print("LoRA enabled")
                self.model.print_trainable_parameters()

        self.accelerator = Accelerator(
            project_dir=self.args.output_path, log_with="wandb"
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.init_lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.args.max_lr,
            total_steps= self.accelerator.num_processes * self.args.epochs * len(self.train_dataloader)
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optimizer, 
        #     T_0=1, T_mult=2
        # )
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        ### Accelerator Init ###


        self.experiment_config = {
            "epochs": self.args.epochs,
            "batch_size": self.args.batch_size,
            "init_lr": self.args.init_lr,
            "max_lr": self.args.max_lr,
            "mode": self.args.mode,
            "lora": self.args.lora,
            "scheduler": str(self.scheduler),
            "optimizer": "Adam"
        }
        self.accelerator.init_trackers(
            project_name=WANDB_PROJECT, 
            config=self.experiment_config, 
            # init_kwargs={"wandb":{"name": f"{self.args.test_name}"}}
        )

        self.device = self.accelerator.device
        self.model, self.optimizer, self.train_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, 
            self.optimizer, 
            self.train_dataloader,
            self.scheduler
        )
        
        self.accelerator.register_for_checkpointing(self.scheduler)

        self.epoch = 0
        if self.args.from_checkpoint is not None:
            self.accelerator.print(f"Resuming from Checkpoint: {self.args.from_checkpoint}")
            self.epoch = int(self.args.from_checkpoint.split("_")[-2])
            self.accelerator.load_state(self.args.from_checkpoint)

        print("Model initialized, training mode: ", self.args.mode)

        self.criterion = torch.nn.CrossEntropyLoss()

        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)
            
        self.log_file = open(
            os.path.join(self.args.output_path, f"{self.args.test_name}_log.txt"),
            "w",
        )
        self.log_file.write(f"CropType training, mode: {self.args.mode}\n")
        self.log_file.write(f"Config: {str(self.args)}\n")

    def load_datasets(self):
        train_ds = CropTypeDataset(
            path=self.args.data_path,
            subset="train",
            bands=None,
            seq_len=self.args.seq_len,
            include_masks=self.args.masked,
        )
        val_ds = CropTypeDataset(
            path=self.args.data_path,
            subset="val",
            bands=None,
            seq_len=self.args.seq_len,
            include_masks=self.args.masked,
        )
        test_ds = CropTypeDataset(
            path=self.args.data_path,
            subset="test",
            bands=None,
            seq_len=self.args.seq_len,
            include_masks=self.args.masked,
        )

        return train_ds, val_ds, test_ds

    def initialize_model(self):
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                "task_name": "classification",
                "n_channels": args.bands,
                "num_class": len(self.test_dataloader.dataset.labels[0]),
                "freeze_encoder": False
                if self.args.mode == "full_finetuning"
                else True,
                "freeze_embedder": False
                if self.args.mode == "full_finetuning"
                else True,
                "reduction": self.args.reduction,
                "enable_gradient_checkpointing": False
            },
        )

        model.init()

        return model

    def train(self):
        for epoch in range(self.args.epochs):
            self.epoch += 1

            self.train_epoch()
            self.evaluate_epoch(phase="val")

    def train_epoch(self):
        """
        Train encoder and classification head (with accelerate enabled)
        """
        self.model.to(self.device)
        self.model.train()
        total_loss, total_correct = 0, 0

        for batch in tqdm(
            self.train_dataloader,
            total=len(self.train_dataloader),
            disable= not self.accelerator.is_local_main_process,
        ):
            self.optimizer.zero_grad()

            if self.args.masked:
                batch_x, batch_labels, batch_masks = batch
                batch_masks = batch_masks.to(self.device)
            else:
                batch_x, batch_labels = batch
                batch_masks = None

            batch_x = batch_x.to(self.device).float()
            batch_labels = batch_labels.to(self.device)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16
                if torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 8
                else torch.float32,
            ):
                output = self.model(batch_x, input_mask=batch_masks, reduction=self.args.reduction)
                loss = self.criterion(output.logits, batch_labels)

            total_loss += loss.item()
            total_correct += (
                (output.logits.argmax(dim=1) == batch_labels.argmax(dim=1))
                .sum()
                .item()
            )

            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()

        avg_loss = total_loss / len(self.train_dataloader)
        
        ### Log training metrics ###
        if self.accelerator.is_main_process:
            self.log_metrics(avg_loss, None, phase='train')


#####################################evaluate loops#################################################
    def test(self):
        self.evaluate_epoch(phase="test")

    def evaluate_epoch(self, phase="val", masked=False):
        if phase == "val":
            dataloader = self.val_dataloader
        elif phase == "test":
            dataloader = self.test_dataloader
        else:
            raise ValueError("Invalid phase, please choose val or test")

        self.model.eval()
        self.model.to(self.device)
        total_loss, total_correct = 0, 0

        with torch.no_grad():
            for batch in tqdm(
                dataloader,
                total=len(dataloader),
                disable=not self.accelerator.is_local_main_process,
            ):
                if self.args.masked:
                    batch_x, batch_labels, batch_masks = batch
                    batch_masks = batch_masks.to(self.device)
                else:
                    batch_x, batch_labels = batch
                    batch_masks = None

                batch_x = batch_x.to(self.device).float()
                batch_labels = batch_labels.to(self.device)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16
                    if torch.cuda.is_available()
                    and torch.cuda.get_device_capability()[0] >= 8
                    else torch.float32,
                ):
                    output = self.model(batch_x, input_mask=batch_masks)
                    loss = self.criterion(output.logits, batch_labels)
                    
                total_loss += loss.item()
                total_correct += (
                    (output.logits.argmax(dim=1) == batch_labels.argmax(dim=1))
                    .sum()
                    .item()
                )

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / len(dataloader.dataset)
        
        if self.accelerator.is_main_process:
            self.log_metrics(avg_loss, accuracy, phase=phase)

#################################Logging and Checkpoints###########################################
    def log_metrics(self, loss, accuracy, phase="train"):

        if phase == "train":
            self.accelerator.print(
                "Phase: {phase}, Epoch {self.epoch}, Loss: {loss}"
            )
            self.accelerator.log(
                {
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "training_loss": loss,
                },
                step=self.epoch,
            )
            self.log_file.write(
                f"Phase: {phase}, Epoch {self.epoch}, Loss: {loss}\n"
            )
        else:
            self.accelerator.print(
                f"Phase: {phase}, Epoch {self.epoch}, Loss: {loss}, Accuracy: {accuracy}"
            )
            self.accelerator.log(
                {f"{phase}_loss": loss, f"{phase}_accuracy": accuracy},
                step=self.epoch,
            )
            self.log_file.write(
                f"Phase: {phase}, Epoch {self.epoch}, Loss: {loss}, Accuracy: {accuracy}\n"
            )

    def save_checkpoint(self):
        path = self.args.output_path

        # mkdir if not exist
        if not os.path.exists(path):
            os.makedirs(path)

        # save parameter that requires grad
        self.accelerator.save_state(
            output_dir=os.path.join(path, f"{self.args.test_name}_{self.epoch}_checkpoint")
        )
        print("Model saved at ", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--mode", type=str, default="full_finetuning", help="choose from linear_probing, full_finetuning")
    parser.add_argument("--init_lr", type=float, default=1e-6)
    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora", action="store_true", help="enable LoRA")
    parser.add_argument("--reduction", type=str, default="concat", help="reduction method for MOMENT embeddings, choose from mean, concat, or max")
    parser.add_argument("--data_path", type=str, help="path to crop dataset")
    parser.add_argument("--output_path", type=str, help="path to save trained model and logs")
    parser.add_argument("--seq_len", type=int, default=512, help="sequence length for each sample, currently only support 512 for MOMENT")
    parser.add_argument("--bands", type=int, default=2, help="number of bands for input data")
    parser.add_argument("--test_name", type=str, default="moment_classification", help="name of the test. will be used as checkpoint name")
    parser.add_argument("--from_checkpoint", type=str, default=None, help="path to model checkpoint")
    parser.add_argument("--masked", type=bool, default=False, help="train on masked")

    args = parser.parse_args()
    control_randomness(args.seed)
    
    trainer = CropTypeTrainer(args)
    trainer.train()
    trainer.test()
    trainer.save_checkpoint()
