import os
import torch
from torch.utils.data import DataLoader, random_split
import hashlib
import base64
from loguru import logger
from anomaly_detection.utils.models import get_model
from anomaly_detection.data.dataset import HDFSEventsDataset
from anomaly_detection.data.hdfs_series import HDFSEvents
from anomaly_detection.utils.paths import main_repo, checkpoints_folder, logs_folder
from anomaly_detection.utils.configs import training_configs
from anomaly_detection.trainers.trainer import Trainer

def load_and_prepare_data(config):
    """
    Loads the dataset, performs splitting into training and validation sets,
    and prepares DataLoader objects for both.

    Args:
        config (dict): Training configuration.

    Returns:
        tuple: Tuple containing training and validation DataLoader objects.
    """
    events_file_path = main_repo / 'data/hdfs/hdfs_train'
    events = HDFSEvents.from_text_file(events_file_path, nrows=100)
    dataset = HDFSEventsDataset(events, window_size=config["window_size"])

    # Split dataset into training and validation sets
    total_size = len(dataset)
    train_size = int(total_size * config["train_val_split"])
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Setup DataLoader for both training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader

def setup_and_train(model_name: str, model_variant: str, training_config_name: str):
    """
    Setup and train the model based on the specified names and configurations.

    Args:
        model_name (str): Name of the model.
        model_variant (str): Variant of the model.
        training_config_name (str): Name of the training configuration.
    """
    if os.path.exists("debug.log"):
        os.remove("debug.log")
    logger.remove()
    logger.add("debug.log", format="{function}:{line} - {message}", level="INFO")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    model = get_model(model_name, model_variant).to(device)
    logger.info(f"{model_name} model of variant '{model_variant}' loaded successfully.")

    config = training_configs[training_config_name]

    config_str = str(sorted(config.items()))
    hash_obj = hashlib.sha256(config_str.encode())
    config_chars = base64.urlsafe_b64encode(hash_obj.digest()).decode('utf-8')[:5]
    config_id = f"{model_name}_{model_variant}_{config_chars}"
    logger.info(f"Configuration ID: {config_id}")

    train_loader, val_loader = load_and_prepare_data(config)

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.functional.cross_entropy

    trainer = Trainer(train_loader, val_loader, model, optimizer, loss_fn,
                      checkpoint_dir=checkpoints_folder / config_id,
                      log_dir=logs_folder / config_id)
    trainer.train(max_iters=config["max_iters"], 
                  eval_interval=config["eval_interval"],
                  checkpoint_interval=config["checkpoint_interval"])

# Example usage
setup_and_train("DeepLog", "original", "config_1")
