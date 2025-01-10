from typing import Any
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import numpy as np
import lightning as L
import sklearn.metrics as metrics
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix

class JointNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, classify) -> None:
        super().__init__()
        if classify:
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
                nn.Softmax(dim=1)
            )
        else:
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )
    
    def forward(self, x):
        return self.network(x)

class LitSpot(L.LightningModule):
    def __init__(self, input_dim: int, output_dim: int, markers_path, classify, seq) -> None:
        super().__init__()
        self.model = JointNetwork(input_dim, output_dim, classify)
        self.seq = seq
        # self.device = device  

        self.classify = classify
        self.learning_rate = 2e-3

        markers_pos = np.loadtxt(markers_path, delimiter=',')
        self.marker_positions = {f"{i}": pos for i, pos in enumerate(markers_pos)}
        self.marker_positions['100'] = np.array([0, 0, 0])

        self.rev_marker_positions = {
            tuple(np.round(v.astype(np.float32), decimals=4)): k for k, v in self.marker_positions.items()}

        self.marker_posarray = np.array(list(self.marker_positions.values()))
    
    def forward(self, x):
        return self.model(x)

    def _get_device(self):
        return next(self.parameters()).device

    def training_step(self, batch, batch_idx):
        device = self._get_device()
        x, y = batch["joint_data"].float().to(device), batch["contact_label"].float().to(device)
        
        y_hat = self(x) # shape batch_size by 3

        threshold = 0.1 * np.sqrt(self.seq)
        if self.classify:
            y_hat_idx = torch.argmax(y_hat, dim=1)
            y_idx = torch.argmax(y, dim=1)
            loss = F.cross_entropy(y_hat, y_idx)

            y_hat_label = y_hat_idx.float()         # Convert to float if needed
            y_label = y_idx.float()

            y_hat_pos = np.array([self.marker_positions.get(str(i.item())) for i in y_hat_idx])
            y_pos = np.array([self.marker_positions.get(str(i.item())) for i in y_idx]) 
            
            correct = np.linalg.norm(y_pos - y_hat_pos, axis=1) < threshold
            acc = np.mean(correct)
            euclidean_distance = np.linalg.norm(y_pos - y_hat_pos, axis=1) / np.sqrt(self.seq)

        else:
            loss = F.mse_loss(y_hat, y)
            y_hat_pos = y_hat.detach().cpu().numpy()
            y_pos = y.cpu().numpy()

            euclidean_distance = np.linalg.norm(y_pos - y_hat_pos, axis=1) / np.sqrt(self.seq)

            correct = np.linalg.norm(y_pos - y_hat_pos, axis=1) < threshold
            acc = np.mean(correct)

        self.log("train_loss", loss, on_epoch = True, prog_bar = True)
        self.log("train_acc", acc, on_epoch = True, prog_bar = True)
        self.log("train_dist", euclidean_distance.mean(), on_epoch = True, prog_bar = True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        device = self._get_device()
        x, y = batch["joint_data"].float().to(device), batch["contact_label"].float().to(device)

        y_hat = self(x)

        threshold = 0.1 * np.sqrt(self.seq)
        if self.classify:
            y_hat_idx = torch.argmax(y_hat, dim=1)
            y_idx = torch.argmax(y, dim=1)
            loss = F.cross_entropy(y_hat, y_idx)

            y_hat_label = y_hat_idx.float()         
            y_label = y_idx.float()

            y_hat_pos = np.array([self.marker_positions.get(str(i.item())) for i in y_hat_idx])
            y_pos = np.array([self.marker_positions.get(str(i.item())) for i in y_idx])

            correct = np.linalg.norm(y_pos - y_hat_pos, axis=1) < threshold
            acc = np.mean(correct)

            euclidean_distance = np.linalg.norm(y_pos - y_hat_pos, axis=1) / np.sqrt(self.seq)

        else:
            loss = F.mse_loss(y_hat, y)
            y_hat_pos = y_hat.detach().cpu().numpy()
            y_pos = y.cpu().numpy()

            euclidean_distance = np.linalg.norm(y_pos - y_hat_pos, axis=1) / np.sqrt(self.seq)

            correct = np.linalg.norm(y_pos - y_hat_pos, axis=1) < threshold
            acc = np.mean(correct)

        self.log("val_loss", loss, on_epoch = True, prog_bar = True)
        self.log("val_acc", acc, on_epoch = True, prog_bar = True)
        self.log("val_dist", euclidean_distance.mean(), on_epoch = True, prog_bar = True)

        return loss
    

    def on_validation_epoch_end(self):
        avg_val_loss = self.trainer.logged_metrics.get("val_loss")
        val_acc = self.trainer.logged_metrics.get("val_acc")
        val_dist = self.trainer.logged_metrics.get("val_dist")
        print(f"Epoch {self.current_epoch}: Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Euclidean Distance: {val_dist:.4f}")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def predict(self, inputs):
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            return self(inputs)