from dataclasses import dataclass
import json
from typing import Any, List

def get_config_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_model_config(config_path):
    args = get_config_data(config_path)
    model_args = ModelArgs(**args)
    return model_args