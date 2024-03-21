from pathlib import Path


main_repo = Path(__file__).parent.parent.parent
output_folder = main_repo / "output" / "models"
models_config_path = main_repo / "anomaly_detection" / "configs" / "models_configs.toml"
training_config_path = main_repo / "anomaly_detection" / "configs" / "training_configs.toml"

output_folder = main_repo / "output"
checkpoints_folder = output_folder / "checkpoints" 
logs_folder = output_folder / "logs"
models_folder = output_folder / "models"