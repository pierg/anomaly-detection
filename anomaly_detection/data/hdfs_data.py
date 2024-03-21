from pathlib import Path
from torch.utils.data import DataLoader, random_split
from typing import Tuple

from anomaly_detection.data.dataset import HDFSEventsDataset
from anomaly_detection.data.hdfs_series import HDFSEvents

def load_hdfs_events(hdfs_data_path: Path, file_subpath: str, config: dict) -> DataLoader:
    """
    Load HDFS events from a text file, create a dataset, and return a DataLoader.
    
    :param hdfs_data_path: Base path for HDFS data files.
    :param file_subpath: Subpath for the specific HDFS file.
    :param config: Configuration dictionary with settings for loading and processing data.
    :return: DataLoader for the created dataset.
    """
    file_path = hdfs_data_path / file_subpath.strip("/")
    events = HDFSEvents.from_text_file(file_path, nrows=100)
    dataset = HDFSEventsDataset(events, window_size=config["window_size"])
    return DataLoader(dataset, batch_size=config["batch_size"])

def load_and_prepare_hdfs_data(hdfs_data_path: Path, config: dict) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Load data, create datasets, and split into training, validation, and test dataloaders.
    
    :param hdfs_data_path: Path object pointing to the base directory of HDFS data.
    :param config: Configuration dictionary with settings like window size, batch size, and train-validation split ratio.
    :return: A tuple containing train_loader, val_loader, test_normal_loader, and test_abnormal_loader.
    """
    # Load and prepare training and validation data
    train_val_loader = load_hdfs_events(hdfs_data_path, 'hdfs_train', config)
    total_size = len(train_val_loader.dataset)
    train_size = int(total_size * config["train_val_split"])
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(train_val_loader.dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Load test data
    test_normal_loader = load_hdfs_events(hdfs_data_path, 'hdfs_test_normal', config)
    test_abnormal_loader = load_hdfs_events(hdfs_data_path, 'hdfs_test_abnormal', config)

    return train_loader, val_loader, test_normal_loader, test_abnormal_loader
