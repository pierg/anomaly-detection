"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import base64
import hashlib


def get_config_id(model_name: str, model_variant: str, config: dict) -> str:
    config_str = str(sorted(config.items()))
    hash_obj = hashlib.sha256(config_str.encode())
    config_chars = base64.urlsafe_b64encode(hash_obj.digest()).decode("utf-8")[:5]
    config_id = f"{model_name}_{model_variant}_{config_chars}"
    return config_id
