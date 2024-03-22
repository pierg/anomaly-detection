"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

from pathlib import Path
from torch.utils.data import DataLoader, random_split, Dataset
import torch
from typing import Tuple

from anomaly_detection.data.hdfs_dataset import HDFSEventsDataset
from anomaly_detection.data.hdfs_series import HDFSEvents


from torch.utils.data import Dataset
import torch
from data.hdfs_series import HDFSEvents


def load_hdfs_events(hdfs_data_path: Path, file_subpath: str, config: dict, device: str) -> DataLoader:
    """
    Load HDFS events from a text file, create a dataset, and return a DataLoader.
    
    :param hdfs_data_path: Base path for HDFS data files.
    :param file_subpath: Subpath for the specific HDFS file.
    :param config: Configuration dictionary with settings for loading and processing data.
    :return: DataLoader for the created dataset.
    """
    file_path = hdfs_data_path / file_subpath.strip("/")
    events = HDFSEvents.from_text_file(file_path, nrows=config["nrows"])
    dataset = HDFSEventsDataset(events, window_size=config["window_size"], device=device)
    return DataLoader(dataset, batch_size=config["batch_size"], collate_fn=lambda x: custom_collate_fn(x, config["window_size"], config["num_features"]))

def custom_collate_fn(batch, window_size, num_features):
    """
    Custom collate function to transform and batch data.
    """
    batch_x = [item[0] for item in batch]
    batch_y = [item[1] for item in batch]
    
    batch_x_transformed = torch.stack([x.view(window_size, num_features).float() for x in batch_x])

    return batch_x_transformed, torch.tensor(batch_y, device=batch_x_transformed.device)

def load_and_prepare_hdfs_data(hdfs_data_path: Path, config: dict, device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Load data, create datasets, and split into training, validation, and test dataloaders.
    """
    # Load training and validation data
    train_val_loader = load_hdfs_events(hdfs_data_path, 'hdfs_train', config, device)
    total_size = len(train_val_loader.dataset)
    train_size = int(total_size * config["train_val_split"])
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(train_val_loader.dataset, [train_size, val_size])

    # Applying transformations directly in the DataLoader through a custom collate function
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=lambda x: custom_collate_fn(x, config["window_size"], config["num_features"]))
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=lambda x: custom_collate_fn(x, config["window_size"], config["num_features"]))

    # Load test data
    test_normal_loader = load_hdfs_events(hdfs_data_path, 'hdfs_test_normal', config, device)
    test_abnormal_loader = load_hdfs_events(hdfs_data_path, 'hdfs_test_abnormal', config, device)

    return train_loader, val_loader, test_normal_loader, test_abnormal_loader
