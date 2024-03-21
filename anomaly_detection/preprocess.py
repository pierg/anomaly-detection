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

def setup_and_train(model_name: str, 
                    model_variant: str, 
                    training_config_name: str):
    """
    Setup and train the model based on the specified names and configurations.

    Args:
        model_name (str): Name of the model.
        model_variant (str): Variant of the model.
        training_config_name (str): Name of the training configuration.
    """
    # Clear the log file before starting a new session
    if os.path.exists("debug.log"):
        os.remove("debug.log")

    # Configure loguru logger
    logger.remove()
    logger.add("debug.log", format="{function}:{line} - {message}", level="INFO")

    # Check and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load the model
    model = get_model(model_name, model_variant).to(device)
    logger.info(f"{model_name} model of variant '{model_variant}' loaded successfully.")

    # Load the configuration
    config = training_configs[training_config_name]

    # Generate a unique ID for the configuration
    config_str = str(sorted(config.items()))
    hash_obj = hashlib.sha256(config_str.encode())
    config_chars = base64.urlsafe_b64encode(hash_obj.digest()).decode('utf-8')[:5]
    config_id = f"{model_name}_{model_variant}_{config_chars}"
    logger.info(f"Configuration ID: {config_id}")

    # Load the dataset
    events_file_path = main_repo / 'data/hdfs/hdfs_train'
    events = HDFSEvents.from_text_file(events_file_path, nrows=100)
    dataset = HDFSEventsDataset(events, window_size=config["window_size"])

    # Split dataset into training and validation sets
    total_size = len(dataset)
    train_size = int(total_size * config["train_val_split"])
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Setup DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Setup the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.functional.cross_entropy

    # Initialize and start the trainer
    trainer = Trainer(train_loader, 
                      val_loader, 
                      model, 
                      optimizer, 
                      loss_fn,
                      checkpoint_dir=checkpoints_folder / config_id,
                      log_dir=logs_folder / config_id)
    
    trainer.train(max_iters=config["max_iters"], 
                  eval_interval=config["eval_interval"],
                  checkpoint_interval=config["checkpoint_interval"])
