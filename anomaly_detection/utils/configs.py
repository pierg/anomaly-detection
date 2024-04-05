"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

# from anomaly_detection.models.deeplog import DeepLog
import importlib

import tomlkit

from anomaly_detection.utils.paths import (models_config_path,
                                           training_config_path)


def load_config(file_path):
    with open(file_path, "r") as file:
        return tomlkit.load(file)


models_configs = load_config(models_config_path)

training_configs = load_config(training_config_path)
