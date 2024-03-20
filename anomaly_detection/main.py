"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from anomaly_detection.trainers.trainer import Trainer
from anomaly_detection.utils.tensors import print_batch_info
from data.hdfs_series import HDFSEvents
from data.dataset import HDFSEventsDataset
from models.deeplog import DeepLog
from utils.torch import save_model_info
from loguru import logger

# Configure loguru logger
logger.add("debug.log", level="DEBUG")


def setup_paths():
    main_repo = Path(__file__).parent.parent
    save_folder = main_repo / "output" / "models"
    return save_folder, main_repo

def check_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model(name, config):
    models = {
        "deeplog": lambda config: DeepLog(input_size=config["input_size"], hidden_size=config["hidden_size"], num_layers=config["num_layers"], output_size=config["output_size"]),
        # Add new models here as needed.
    }
    return models[name](config) if name in models else None

def get_hyperparameters(model_name):
    configs = {
        "deeplog": {
            "train_val_split": 0.8,
            "window_size": 10,
            "max_iters": 1000,
            "eval_interval": 100,
            "input_size": 1,
            "hidden_size": 64,
            "num_layers": 2,
            "output_size": 28,
            "batch_size": 16,
        },
    }
    return configs[model_name]

def process_dataset(window_size):
    events = HDFSEvents.from_text_file(Path('data/hdfs/hdfs_train'), nrows=100)
    dataset = HDFSEventsDataset(events, window_size=window_size)
    return dataset

def split_dataset(dataset, train_val_split):
    total_size = len(dataset)
    train_size = int(total_size * train_val_split)
    val_size = total_size - train_size
    return random_split(dataset, [train_size, val_size])

def create_dataloaders(train_dataset, val_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def process_model(model_name="deeplog"):
    save_folder = setup_paths()
    device = check_device()

    hp = get_hyperparameters(model_name)
    model = get_model(model_name, hp).to(device)

    dataset = process_dataset(hp["window_size"])

    train_dataset, val_dataset = split_dataset(dataset, hp["train_val_split"])
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, hp["batch_size"])

    # Print dataset of one data point
    xs, xy = next(iter(train_loader))
    print_batch_info(xs, xy)
    

    # Dummy input for saving model info
    dummy_input = torch.zeros((hp["batch_size"], hp["window_size"], 1), dtype=torch.float32, device=device)
    save_model_info(model, input_tensor=dummy_input, folder=save_folder, id=model_name)

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.functional.cross_entropy

    trainer = Trainer(train_loader, val_loader, model, optimizer, loss_fn)
    trainer.train(max_iters=hp["max_iters"], eval_interval=hp["eval_interval"])

if __name__ == '__main__':
    process_model("deeplog")
