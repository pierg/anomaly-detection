import os
import torch
import torch.nn as nn
from loguru import logger

# Placeholder imports, replace with your actual modules
from anomaly_detection.data.hdfs_data import load_and_prepare_hdfs_data
from anomaly_detection.utils.models import get_model
from anomaly_detection.utils.paths import main_repo, checkpoints_folder, logs_folder
from anomaly_detection.utils.configs import training_configs
from anomaly_detection.trainers.trainer import Trainer
from anomaly_detection.utils.string_utils import get_config_id

# Global settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup logger
if os.path.exists("logs.log"):
    os.remove("logs.log")
logger.remove()
logger.add("logs.log", format="{time:HH:mm:ss} {message}", level="INFO")


configs = ["config_1"] 
models = [("DeepLog", "original")]

def main():

    for config_name in configs:
        config = training_configs[config_name]
        train_loader, val_loader, test_normal_loader, test_abnormal_loader = load_and_prepare_hdfs_data(main_repo / "data/hdfs", config)

        for model_name, model_variant in models:

            config_id = get_config_id(model_name, model_variant, config)

            print(f"Training {config_id} ...")
            
            model = get_model(model_name, model_variant).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            loss_fn = nn.CrossEntropyLoss()
        
            trainer = Trainer(train_loader, val_loader, model, optimizer, loss_fn,
                              checkpoint_dir=checkpoints_folder / config_id,
                              log_dir=logs_folder / config_id)
            
            logger.info(f"Training {model_name} with variant '{model_variant}' using config '{config_name}'")
            trainer.train(max_iters=config["max_iters"], 
                          eval_interval=config["eval_interval"],
                          checkpoint_interval=config["checkpoint_interval"])

            precision, recall, f1_score = trainer.evaluate(test_normal_loader, 
                             test_abnormal_loader, 
                             num_candidates=config["num_candidates"])
            
            print(f"Training and evaluation completed for {config_id}")
            print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

            logger.info(f"Completed training and evaluation for {model_name} variant '{model_variant}' with config '{config_name}'")

if __name__ == "__main__":
    main()

