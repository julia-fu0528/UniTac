import argparse
import os
import sys
import numpy as np
from natsort import natsorted
from pathlib import Path
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from network import LitSpot
from dataset import SpotDataModule

def main(num_classes, markers_path, classify, seq):
    seed_everything(42)

    log_dir = os.path.join(Path(__file__).parent.parent, "gouger_logs")

    if classify:
        tb_logger = TensorBoardLogger(log_dir, name="classification")
    else:
        tb_logger = TensorBoardLogger(log_dir, name="regression")

    data_module = SpotDataModule(classify, seq, batch_size=128)
    if classify:
        output_dim = num_classes
    else:
        output_dim = 3
    model = LitSpot(input_dim = 24 * seq, output_dim = output_dim * seq, markers_path = markers_path, classify=classify, seq=seq)


    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", save_last=True, filename="best")
    early_stop_callback = EarlyStopping(monitor="val_acc", patience=100, mode="max")

    trainer = Trainer(
        # accelerator="gpu",
        accelerator=device,
        # accelerator="cpu",
        max_epochs=20,
        logger=[tb_logger],
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    trainer.fit(model, data_module)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--session', type = str, required=True, help='Session for data collection')
    parser.add_argument('--data_dir', required=True, help='Output directory for data')
    parser.add_argument('--markers_path', required=True, help='Path to markers positions')
    parser.add_argument('--device', required=True, help='gpu or cpu')
    parser.add_argument('--classify', action='store_true', help='Run classification model instead of regression')
    parser.add_argument('--seq', type=int, help='Train on sequence data, length of sequence')


    options = parser.parse_args()

    classify = options.classify
    session = options.session
    data_dir = options.data_dir
    markers_path = options.markers_path
    device = options.device
    seq = options.seq

    # current folder path 
    folder_path =  Path(__file__).parent
    torque_dir = os.path.join(folder_path, data_dir, session)
    # get all the files with .npy extension
    subdirs = natsorted(os.listdir(torque_dir))
    torque_files = [f for f in os.listdir(os.path.join(torque_dir, subdirs[0])) if f.endswith('.npy')]
    torque_files = natsorted(torque_files)
    # get all the file names
    classes = [f.split('.')[0] for f in torque_files]
    num_classes = len(classes)

    # markers_pos = np.loadtxt(markers_path, delimiter=",")
    # marker_positions = {f"{i}": pos for i, pos in enumerate(markers_pos)}

    # calculate the euclidean distance between the 10th and 11th marker
    # marker_10 = marker_positions.get('58')
    # marker_11 = marker_positions.get('61')
    # euclidean_distance = np.linalg.norm(marker_10 - marker_11)
    # print(f"Euclidean distance between marker 10 and 11: {euclidean_distance}")
    # sys.exit()

    main(num_classes, markers_path, classify, seq)
    