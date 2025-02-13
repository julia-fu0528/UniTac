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
                nn.Dropout(0.3),

                nn.Linear(128, 128),
                nn.ReLU(),

                nn.Linear(128, 128),
                nn.ReLU(),

                nn.Linear(128, output_dim),
                nn.Softmax(dim=1)
            )
        else:
            # spot
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 64),
                nn.ReLU(), 

                nn.Linear(64,128), 
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                # nn.Linear(256, 256),
                # nn.ReLU(),
                # nn.Dropout(0.3),

                # nn.Linear(256, 128),
                # nn.ReLU(),
                # nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(128, output_dim)
                
            )
            # franka
            # self.network = nn.Sequential(
            #     nn.Flatten(),
            #     nn.Linear(input_dim, 64),
            #     nn.ReLU(), 

            #     nn.Linear(64,128), 
            #     nn.ReLU(),
            #     nn.Dropout(0.3),
            #     nn.Linear(128, 256),
            #     nn.ReLU(),
            #     nn.Dropout(0.3),
            #     nn.Linear(256, 128),
            #     nn.ReLU(),
            #     nn.Dropout(0.3),

            #     nn.Linear(128, output_dim)
                
            # )
    
    def forward(self, x):

        return self.network(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return F.relu(x + self.layers(x))

class LitRobot(L.LightningModule):
    def __init__(self, input_dim: int, output_dim: int, markers_path, classify, seq, robot_type) -> None:
        super().__init__()
        self.model = JointNetwork(input_dim, output_dim, classify)
        self.seq = seq
        self.test_outputs = {"preds": [], "labels": []}
        self.robot_type = robot_type
        # self.device = device   

        self.classify = classify
        self.learning_rate = 2e-3

        markers_pos = np.loadtxt(markers_path, delimiter=',')
        self.marker_positions = {f"{i}": pos for i, pos in enumerate(markers_pos)}
        self.marker_positions[str(len(markers_pos))] = np.array([0, 0, 0])

        self.rev_marker_positions = {
            tuple(np.round(v.astype(np.float32), decimals=4)): k for k, v in self.marker_positions.items()}

        self.marker_posarray = np.array(list(self.marker_positions.values()))
    
    def forward(self, x):
        return self.model(x)

    def _get_device(self):
        return next(self.parameters()).device

    def normalize_input(self, x):
        return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

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
            
            # if self.robot_type == 'franka':
            #     acc = metrics.accuracy_score(y_idx.cpu().numpy(), y_hat_idx.cpu().numpy())
            # elif self.robot_type == 'spot':
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

        self.log("train/loss", loss, on_epoch = True, prog_bar = True)
        self.log("train/acc", acc, on_epoch = True, prog_bar = True)
        self.log("train/dist", euclidean_distance.mean(), on_epoch = True, prog_bar = True)

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

            # if self.robot_type == 'franka':
            #     acc = metrics.accuracy_score(y_idx.cpu().numpy(), y_hat_idx.cpu().numpy())
            # elif self.robot_type == 'spot':
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

        self.log("val/loss", loss, on_epoch = True, prog_bar = True)
        self.log("val/acc", acc, on_epoch = True, prog_bar = True)
        self.log("val/dist", euclidean_distance.mean(), on_epoch = True, prog_bar = True)


        return loss

    def test_step(self, batch, batch_idx):
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
            # if self.robot_type == 'franka':
            #     acc = metrics.accuracy_score(y_idx.cpu().numpy(), y_hat_idx.cpu().numpy())
            # elif self.robot_type == 'spot':
            acc = np.mean(correct)

            euclidean_distance = np.linalg.norm(y_pos - y_hat_pos, axis=1) / np.sqrt(self.seq)

        else:
            loss = F.mse_loss(y_hat, y)
            y_hat_pos = y_hat.detach().cpu().numpy()
            y_pos = y.cpu().numpy()

            euclidean_distance = np.linalg.norm(y_pos - y_hat_pos, axis=1) / np.sqrt(self.seq)
            
            # calculate the distance from y_hat_pos to all the marker positions
            distances = np.linalg.norm(self.marker_posarray[None, :, :] - y_hat_pos[:, None, :], axis=2)
            min_indices = np.argmin(distances, axis=1)
            min_indices = torch.tensor(min_indices, dtype=torch.long).to(device)
            y_hat_label = min_indices

            y_label = np.zeros(y_pos.shape[0])
            for i, pos in enumerate(y_pos):
                rounded_pos = tuple(np.round(pos, decimals=4))
                y_label[i] = int(self.rev_marker_positions.get(rounded_pos))
            y_label = torch.tensor(y_label, dtype=torch.long).to(device)

            correct = np.linalg.norm(y_pos - y_hat_pos, axis=1) < threshold
            acc = np.mean(correct)

        self.log("test/loss", loss, on_epoch = True, prog_bar = True)
        self.log("test/acc", acc, on_epoch = True, prog_bar = True)
        self.log("test/dist", euclidean_distance.mean(), on_epoch = True, prog_bar = True)

        # if self.classify:
        self.test_outputs["preds"].append(y_hat_label)
        self.test_outputs["labels"].append(y_label)

        return loss
    

    def on_validation_epoch_end(self):
        avg_val_loss = self.trainer.logged_metrics.get("val/loss")
        val_acc = self.trainer.logged_metrics.get("val/acc")
        val_dist = self.trainer.logged_metrics.get("val/dist")

        print(f"Epoch {self.current_epoch}: Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Euclidean Distance: {val_dist:.4f}")


    def on_test_epoch_end(self):
        # if self.classify:
        all_preds = torch.cat(self.test_outputs["preds"], dim=0)
        print(f"all_preds: {all_preds.shape}")
        all_labels = torch.cat(self.test_outputs["labels"], dim=0)
        print(F"all_labels: {all_labels.shape}")

        all_preds_np = all_preds.cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()

        conf_matrix = metrics.confusion_matrix(all_labels_np, all_preds_np, labels=range(len(self.marker_positions.keys())))

        f1 = f1_score(all_labels_np, all_preds_np, average='weighted')

        self.log('test_f1_score', f1)
        self.plot_confusion_matrix(conf_matrix, class_names = self.marker_positions.keys())
        print(f"Test F1 Score: {f1:.4f}")


        avg_test_loss = self.trainer.logged_metrics.get("test/loss")
        test_acc = self.trainer.logged_metrics.get("test/acc")
        test_dist = self.trainer.logged_metrics.get("test/dist")

        print(f"Epoch {self.current_epoch}: Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Euclidean Distance: {test_dist:.4f}")
    

    def plot_confusion_matrix(self, cm, class_names, file_path="confusion_matrix.png"):
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        model_type = "Classification" if self.classify else "Regression"
        ax.set_title(f'Confusion Matrix {model_type}')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.savefig(file_path)
        plt.close()
        # plt.show()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": scheduler,
        #     "monitor": "val/loss"
        # }
    
    def predict(self, inputs):
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            return self(inputs)