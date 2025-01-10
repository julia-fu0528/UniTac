import json
import os
import sys
import random
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pathlib import Path
import argparse
from natsort import natsorted
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import lightning as L
import torch.nn.functional as F
from argparse import ArgumentParser
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from src.utils.visualize_robot_state import load_joint_torques, load_joint_positions

class JointLabel:
    def __init__(self, torque_dir, markers_path, classify) -> None:
        self.torque_dir = torque_dir
        self.training_data = []
        self.training_labels = []
        self.validation_data = []
        self.validation_labels = []
        
        markers_pos = np.loadtxt(markers_path, delimiter=',')
        self.marker_positions = {f"{i}": pos for i, pos in enumerate(markers_pos)}
        # get all the files with .npy extension
        subdirs = natsorted(os.listdir(torque_dir))
        torque_files = [f for f in os.listdir(os.path.join(torque_dir, subdirs[0])) if f.endswith('.npy')]
        torque_files = natsorted(torque_files)
        # get all the file names
        classes = [f.split('.')[0] for f in torque_files]

        self.classes = classes

        self.num_classes = len(classes)
        coordinates = {}
        for c in classes:
            if self.marker_positions.get(c) is None:
                coordinates['100'] = np.array([0, 0, 0])
            else:
                coordinates[c] = self.marker_positions.get(c)

        self.classify = classify

        if self.classify:
            self.class_to_label = {cls: idx for idx, cls in enumerate(classes)}
        else:
            self.class_to_label = coordinates
        
        self.preprocess_data()
        
    def preprocess_data(self):
        print(f"Preprocessing data...")
        num_dir = len([name for name in os.listdir(self.torque_dir) if os.path.isdir(os.path.join(self.torque_dir, name))])
        start_dir = 28
        end_dir = 39
        # randomly pick a number from 0 to num_dir
        val_indices = random.sample(range(start_dir, end_dir), 2) + random.sample(range(0, 10), 2)
        train_dirs = []
        val_dirs = []
        dirs = natsorted(os.listdir(self.torque_dir))
        for idx, dir in enumerate(dirs):
            if os.path.isdir(os.path.join(self.torque_dir, dir)):
                if idx in val_indices:
                    val_dirs.append(os.path.join(self.torque_dir, dir))
                else:
                    train_dirs.append(os.path.join(self.torque_dir, dir))
        print(f"val dirs: {val_dirs}")
        for train_dir in train_dirs:
            self.load_data(train_dir, mode='train')
        for val_dir in val_dirs:
            self.load_data(val_dir, mode='val')
        print(f"Finished data preprocessing")
    
    def load_data(self, dir, mode='train'):
        print(f"Loading data from {dir} for {mode}...")
        for i, torque_file in enumerate(natsorted(os.listdir(dir))):
            if torque_file.endswith(".npy"):
                torque_path = os.path.join(dir, torque_file)
                class_name = torque_file.split(".")[0] # extract label from the file name
                if class_name == 'no_contact':
                    if not self.classify:
                        class_name = '100'
                if class_name in self.class_to_label.keys():
                    label = self.class_to_label[class_name]
                    torque, _, _, _ = load_joint_torques(torque_path)
                    joint_angle, _, _, _ = load_joint_positions(torque_path)
                    data = np.hstack((torque, joint_angle))
                    normalized_data = 2 * ((data - data.min()) / (data.max() - data.min())) - 1
                    for joint_data in normalized_data:
                        if mode == 'train':
                            self.training_data.append(joint_data)
                        else:
                            self.validation_data.append(joint_data)
                    if self.classify:
                        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes).numpy()
                    for i in range(len(normalized_data)):
                        if mode == 'train':
                            self.training_labels.append(label)
                        else:
                            self.validation_labels.append(label)
        print(f"Finished loading data from {dir}")

class SpotDataset(Dataset):
    def __init__(self, dataset_mode, seq, mode='train') -> None:
        super().__init__()
        self.mode = mode
        self.seq = seq
        self.dataset_mode = dataset_mode
        folder_path = Path(__file__).parent.parent
        self.joint_data = np.load(f"{folder_path}/preprocessed_data/{self.dataset_mode}/{mode}_joint_data_{seq}.npy", allow_pickle=True)
        self.contact_labels = np.load(f"{folder_path}/preprocessed_data/{self.dataset_mode}/{mode}_contact_labels_{seq}.npy", allow_pickle=True)

        assert len(self.joint_data) == len(self.contact_labels), "Length of joint data and contact labels should be the same"
    
    def __len__(self):
        return len(self.joint_data)
    
    def __getitem__(self, idx):
        if self.seq!= 1:
            if idx + self.seq > len(self.joint_data):
                joint_data = np.zeros((self.seq, self.joint_data.shape[1]))
                contact_label = np.zeros((self.seq, self.contact_labels.shape[1]))
                for i in range(self.seq):
                    if idx + i < len(self.joint_data):
                        joint_data[i] = self.joint_data[idx + i]
                        contact_label[i] = self.contact_labels[idx + i]
                    else:
                        joint_data[i] = self.joint_data[-1]
                        contact_label[i] = self.contact_labels[-1]
            else:
                joint_data = self.joint_data[idx:idx+self.seq]
                contact_label = self.contact_labels[idx:idx+self.seq]
            # concantenate
            joint_data = np.concatenate(joint_data, axis=0)
            contact_label = np.concatenate(contact_label, axis=0)
        else:
            joint_data = self.joint_data[idx]
            contact_label = self.contact_labels[idx]   

        return {
            "joint_data": joint_data,
            "contact_label": contact_label
        }

class SpotDataModule(L.LightningDataModule):
    def __init__(self, classify, seq, batch_size=32) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.classify = classify
        self.seq = seq

    def setup(self, stage=None):
        if self.classify:
            self.train_set = SpotDataset(dataset_mode="classify", seq = self.seq,  mode='train')
            self.val_set = SpotDataset(dataset_mode="classify",  seq = self.seq,  mode='val')
        else:
            self.train_set = SpotDataset(dataset_mode="regression",  seq = self.seq,  mode='train')
            self.val_set = SpotDataset(dataset_mode="regression",  seq = self.seq,  mode='val')
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=4,
            pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=4,
            pin_memory=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--session', type = str, required=True, help='Session for data collection')
    parser.add_argument('--data_dir', required=True, help='Output directory for data')
    parser.add_argument('--markers_path', required=True, help='Path to markers positions')
    parser.add_argument('--classify', action='store_true', help='Run classification model instead of regression')
    parser.add_argument('--seq', type=int, help='Train on sequence data, length of sequence')

    options = parser.parse_args()

    classify = options.classify
    seq = options.seq

    session = options.session
    data_dir = options.data_dir
    markers_path = options.markers_path

    markers_pos = np.loadtxt(markers_path, delimiter=",")
    marker_positions = {f"{i}": pos for i, pos in enumerate(markers_pos)}

    # current folder path 
    folder_path =  Path(__file__).parent
    torque_dir = os.path.join(folder_path, data_dir, session)

    joint_label = JointLabel(torque_dir, markers_path, classify=classify)
    
    print(f"Saving training and validation data")
    if classify:
        dataset_mode = "classify"
    else:
        dataset_mode = "regression"

    save_dir = os.path.join(folder_path.parent, "preprocessed_data", dataset_mode)
    os.makedirs(save_dir, exist_ok=True)
    if classify:
        training_labels = np.stack(joint_label.training_labels)  # Stack instead of simple array conversion
        validation_labels = np.stack(joint_label.validation_labels)
    else:
        training_labels = np.array(joint_label.training_labels)
        validation_labels = np.array(joint_label.validation_labels)
    np.save(os.path.join(save_dir,f"train_joint_data_{seq}.npy"), joint_label.training_data)
    np.save(os.path.join(save_dir,f"train_contact_labels_{seq}.npy"), training_labels)
    np.save(os.path.join(save_dir,f"val_joint_data_{seq}.npy"), joint_label.validation_data)
    np.save(os.path.join(save_dir,f"val_contact_labels_{seq}.npy"), validation_labels)
    print(f"Training and validation data saved to {save_dir}")

    train_dataset = SpotDataset(dataset_mode, seq = seq, mode='train')
    print(train_dataset[0])
