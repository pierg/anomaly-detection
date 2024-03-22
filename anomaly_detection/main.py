"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import os
import torch
import torch.nn as nn
from loguru import logger

# Placeholder imports, replace with your actual modules
from anomaly_detection.data.hdfs_data import load_and_prepare_hdfs_data
from anomaly_detection.utils.io_utils import save_results
from anomaly_detection.utils.models import get_model
from anomaly_detection.utils.paths import hdfs_deeplog_data_path, checkpoints_folder, logs_folder
from anomaly_detection.utils.configs import training_configs
from anomaly_detection.trainers.trainer import Trainer
from anomaly_detection.utils.string_utils import get_config_id

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configure logger
logger_config_path = "logs.log"
if os.path.exists(logger_config_path):
    os.remove(logger_config_path)
logger.remove()
logger.add(logger_config_path, format="{time:HH:mm:ss} {level} {message}", level="INFO")

# Configurations and models
configs = ["config_1"]
models = [
            ("DeepLog", "original"), 
            ("EnhancedDL", "original"), 
            ("TransformerDL", "original")
            ]

def main():
    for config_name in configs:
        config = training_configs[config_name]
        data_loaders = load_and_prepare_hdfs_data(hdfs_deeplog_data_path, config)
        train_loader, val_loader, test_normal_loader, test_abnormal_loader = data_loaders

        for model_name, model_variant in models:
            config_id = get_config_id(model_name, model_variant, config)

            logger.info(f"Training {config_id} ...")
            
            model = get_model(model_name, model_variant).to(device)
            # dummy_input = next(iter(train_loader))[0].to(device)
            # save_model_info(model, dummy_input, models_folder / f"{config_id}")

            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            loss_fn = nn.CrossEntropyLoss()

            trainer = Trainer(train_loader, val_loader, model, optimizer, loss_fn,
                              checkpoint_dir=checkpoints_folder / config_id,
                              log_dir=logs_folder / config_id)

            trainer.train(max_iters=config["max_iters"], 
                          eval_interval=config["eval_interval"],
                          checkpoint_interval=config["checkpoint_interval"])

            metrics = trainer.evaluate(test_normal_loader, test_abnormal_loader, num_candidates=config["num_candidates"])
            precision, recall, f1_score = metrics

            logger.info(f"Completed training and evaluation for {config_id}")
            print(f"Training and evaluation completed for {config_id}. Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

            # Call the save_results function
            save_results(config_id, precision, recall, f1_score)

if __name__ == "__main__":
    main()
