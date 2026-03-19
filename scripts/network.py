import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import lightning as L
import sklearn.metrics as metrics
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

class UniTacNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, classify) -> None:
        super().__init__()
        if classify:
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
                
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),


                nn.Linear(128, output_dim),
                nn.Softmax(dim=1)
                
            )
        else:
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
                
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),


                nn.Linear(128, output_dim)
                
            )
    
    def forward(self, x):

        return self.network(x)


class LitRobot(L.LightningModule):
    def __init__(self, input_dim: int, output_dim: int, markers_path, classify, seq, robot_type) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.model = UniTacNet(input_dim, output_dim, classify)
        self.seq = seq
        self.test_outputs = {"preds": [], "labels": []}
        self.train_outputs = {"preds": [], "labels": []}
        self.val_outputs = {"preds": [], "labels": []}
        self.robot_type = robot_type


        self.classify = classify
        self.learning_rate = 2.5e-3 

        markers_pos = np.loadtxt(markers_path, delimiter=',')
        self.marker_positions = {f"{i}": pos for i, pos in enumerate(markers_pos)}
        self.marker_positions[str(len(markers_pos))] = np.array([0, 0, 0])

        self.rev_marker_positions = {
            tuple(np.round(v.astype(np.float32), decimals=5)): k for k, v in self.marker_positions.items()}

        self.marker_posarray = np.array(list(self.marker_positions.values()))

        self.buffers = {}

    
    def forward(self, x):
        return self.model(x)

    def _get_device(self):
        return next(self.parameters()).device

    def normalize_input(self, x):
        return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

    def training_step(self, batch, batch_idx):
        device = self._get_device()
        x, y = batch["joint_data"].float().to(device), batch["contact_label"].float().to(device)

        class_nums = batch["class_num"].float().to(device)
        # drop sequence id feature before feeding MLP
        x = x[:, :-1]

        y_hat = self(x) 

        threshold = 0.1 * np.sqrt(self.seq) # can replace with any other threshold
        if self.classify:
            y_hat_idx = torch.argmax(y_hat, dim=1)
            y_idx = torch.argmax(y, dim=1)
            loss = F.cross_entropy(y_hat, y_idx)

            y_hat_label = y_hat_idx.float()        
            y_label = y_idx.float()

            y_hat_pos = np.array([self.marker_positions.get(str(i.item())) for i in y_hat_idx])
            y_pos = np.array([self.marker_positions.get(str(i.item())) for i in y_idx]) 
            
            # accuracy calculated as predictions within a threshold from ground truth
            correct = np.linalg.norm(y_pos - y_hat_pos, axis=1) < threshold
            acc = np.mean(correct)
            euclidean_distance = np.linalg.norm(y_pos - y_hat_pos, axis=1) / np.sqrt(self.seq)

        else:
            loss = F.mse_loss(y_hat, y)
            y_hat_pos = y_hat.detach().cpu().numpy()
            y_pos = y.cpu().numpy()
            y_label = class_nums

            euclidean_distance = np.linalg.norm(y_pos - y_hat_pos, axis=1) / np.sqrt(self.seq)

            # calculate the distance from y_hat_pos to all the marker positions
            distances = np.linalg.norm(self.marker_posarray[None, :, :] - y_hat_pos[:, None, :], axis=2)
            min_indices = np.argmin(distances, axis=1)
            min_indices = torch.tensor(min_indices, dtype=torch.long).to(device)
            y_hat_label = min_indices

            correct = np.linalg.norm(y_pos - y_hat_pos, axis=1) < threshold
            acc = np.mean(correct)

        self.log("train/loss", loss, on_epoch = True, prog_bar = True)
        self.log("train/acc", acc, on_epoch = True, prog_bar = True)
        self.log("train/dist", euclidean_distance.mean(), on_epoch = True, prog_bar = True)


        self.train_outputs["preds"].append(y_hat_label)
        self.train_outputs["labels"].append(y_label)

        return loss
    
    def validation_step(self, batch, batch_idx):
        device = self._get_device()
        x, y = batch["joint_data"].float().to(device), batch["contact_label"].float().to(device)
        # last feature is sequence id for EMA grouping
        seq_ids = x[:, -1]
        x = x[:, :-1]
        class_nums = batch["class_num"].float().to(device)
        y_hat = self(x)

        copy_yhat = y_hat.detach().cpu().numpy()
        # apply ema per sequence id
        for i in range(y_hat.shape[0]):
            key = int(seq_ids[i].item())
            if self.buffers.get(key) is None:
                self.buffers[key], self.weights = self.create_buffers(self.seq, alpha=0.1, sliding_win = 60) # 60 for spot, 40 for franka
            self.buffers[key] = np.roll(self.buffers[key], 1, axis=0)
            buffer = self.buffers[key]
            buffer[0:self.seq] = copy_yhat[i:i+self.seq, :]
            y_hat[i:i+self.seq] = torch.tensor(np.dot(self.weights, buffer), dtype=torch.float32).to(y_hat.device) # doesn't work for seq > 1

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

            # calculate the distance from y_hat_pos to all the marker positions
            distances = np.linalg.norm(self.marker_posarray[None, :, :] - y_hat_pos[:, None, :], axis=2)
            min_indices = np.argmin(distances, axis=1)
            min_indices = torch.tensor(min_indices, dtype=torch.long).to(device)
            y_hat_label = min_indices

            y_label = np.zeros(y_pos.shape[0])
            for i, pos in enumerate(y_pos):
                rounded_pos = tuple(np.round(pos, decimals=5))
                y_label[i] = int(self.rev_marker_positions.get(rounded_pos))
            y_label = torch.tensor(y_label, dtype=torch.long).to(device)


            correct = np.linalg.norm(y_pos - y_hat_pos, axis=1) < threshold
            acc = np.mean(correct)

        self.log("val/loss", loss, on_epoch = True, prog_bar = True)
        self.log("val/acc", acc, on_epoch = True, prog_bar = True)
        self.log("val/dist", euclidean_distance.mean(), on_epoch = True, prog_bar = True)

        self.val_outputs["preds"].append(y_hat_label)
        self.val_outputs["labels"].append(y_label)

        return loss

    def test_step(self, batch, batch_idx):
        device = self._get_device()
        x, y = batch["joint_data"].float().to(device), batch["contact_label"].float().to(device)
        class_nums = batch["class_num"].float().to(device)

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
            y_label = class_nums.cpu().numpy()

            euclidean_distance = np.linalg.norm(y_pos - y_hat_pos, axis=1) / np.sqrt(self.seq)
            
            # calculate the distance from y_hat_pos to all the marker positions
            distances = np.linalg.norm(self.marker_posarray[None, :, :] - y_hat_pos[:, None, :], axis=2)
            min_indices = np.argmin(distances, axis=1)
            min_indices = torch.tensor(min_indices, dtype=torch.long).to(device)
            y_hat_label = min_indices

            y_label = np.zeros(y_pos.shape[0])
            for i, pos in enumerate(y_pos):
                rounded_pos = tuple(np.round(pos, decimals=5))
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
        all_labels = torch.cat(self.test_outputs["labels"], dim=0)

        all_preds_np = all_preds.cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()
        for i in range(len(all_labels_np)//10 + 1):
            if 10 * (i+1) < len(all_labels_np):
                chosen_classes = np.array(range(10 * i, 10 * (i + 1)))
            else:
                chosen_classes = np.array(range(len(all_labels_np) - 10, len(all_labels_np)))
            # Create a boolean mask
            mask = np.isin(all_labels_np, chosen_classes)  
            chosen_labels = all_labels_np[mask]
            chosen_preds = all_preds_np[mask]
            conf_matrix = metrics.confusion_matrix(chosen_labels, chosen_preds, labels=chosen_classes)

            class_names= list(self.marker_positions.keys())
            selected_class_names = [class_names[i] for i in chosen_classes]
            self.plot_confusion_matrix(conf_matrix, class_names = selected_class_names, file_path = f"confusion_matrix_{i}.png")
        f1 = f1_score(all_labels_np, all_preds_np, average='weighted')
        self.log('test_f1_score', f1)
        print(f"Test F1 Score: {f1:.4f}")


        avg_test_loss = self.trainer.logged_metrics.get("test/loss")
        test_acc = self.trainer.logged_metrics.get("test/acc")
        test_dist = self.trainer.logged_metrics.get("test/dist")

        print(f"Epoch {self.current_epoch}: Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Euclidean Distance: {test_dist:.4f}")
    

    def plot_confusion_matrix(self, cm, class_names, file_path="confusion_matrix.png"):
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
        num_classes = cm.shape[0]  # Should be square

        # Correct the number of ticks
        # num_classes = 10
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def predict(self, inputs):
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            return self(inputs)

    def create_buffers(self, seq_win, alpha=0.95, sliding_win=3):
        seq_win = self.seq
        self.alpha = alpha
        buffer = np.zeros((sliding_win, self.output_dim))
        weights = np.power((1-alpha), np.arange(sliding_win))
        weights = alpha * weights
        if not self.classify:
            weights = weights / np.sum(weights)
        

        return buffer, weights