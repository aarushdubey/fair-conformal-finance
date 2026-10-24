from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load experiment configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The loaded configuration as a nested dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
