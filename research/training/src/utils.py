"""
Config utilities for Elsworld training pipeline.
Dead-code data loaders (load_gameplay_data, load_gameplay_data_lazy) have been
removed — use elworld.data.npz_loader.load_npz_files() instead.
"""

import yaml
from typing import Dict


def load_config(config_path: str = "config.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_general_config(config: Dict) -> Dict:
    return config.get("general_config", {})


def get_vision_config(config: Dict) -> Dict:
    return config.get("vision_config", {})


def get_memory_config(config: Dict) -> Dict:
    return config.get("memory_config", {})


def get_dreaming_config(config: Dict) -> Dict:
    return config.get("dreaming_config", {})
