import importlib
from anomaly_detection.utils.configs import models_configs

def get_model(name: str, options: str):
    try:
        model_config = models_configs[name][options]
        model_module = importlib.import_module(f"anomaly_detection.models.{name.lower()}")
        model_class = getattr(model_module, name)
        return model_class(**model_config)
    except (ImportError, AttributeError, KeyError) as e:
        print(f"Error: {e}")
        return None
    

    