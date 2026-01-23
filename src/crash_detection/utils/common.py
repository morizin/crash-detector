from typeguard import typechecked
from pathlib import Path
from box import ConfigBox
from typing import Any
from .. import logger
import os
import random
import numpy as np
import torch


@typechecked
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@typechecked
def load_yaml(path: Path | str, boxed=True) -> ConfigBox | dict[str | Any]:
    import yaml

    if isinstance(path, str):
        path = Path(path)
    try:
        with open(path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
        logger.info(f"Successfully Loaded YAML file : {path}")
        if boxed:
            config = ConfigBox(config)
        return config
    except Exception as e:
        logger.error(f"Failed to load YAML file : {path}")
        raise e


@typechecked
def save_yaml(path: Path | str, data: dict[str | Any]):
    import yaml

    if isinstance(path, str):
        path = Path(path)
    try:
        with open(path, "w") as yaml_file:
            yaml.dump(data, yaml_file)
        logger.info(f"Successfully saved YAML file : {path}")
    except Exception as e:
        logger.error(f"Failed to save YAML file : {path}")
        raise e


@typechecked
def load_csv(path: Path | str) -> Any:
    import pandas as pd

    if isinstance(path, str):
        path = Path(path)
    try:
        df = pd.read_csv(path)
        logger.info(f"Successfully loaded CSV file : {path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV file : {path}")
        raise e


@typechecked
def save_csv(path: Path | str, data: Any):
    if isinstance(path, str):
        path = Path(path)
    try:
        data.to_csv(path, index=False)
        logger.info(f"Successfully saved CSV file : {path}")
    except Exception as e:
        logger.error(f"Failed to save CSV file : {path}")
        raise e


@typechecked
def load_json(path: Path | str) -> dict[str | Any]:
    import json

    if isinstance(path, str):
        path = Path(path)
    try:
        with open(path, "r") as json_file:
            data = json.load(json_file)
        logger.info(f"Successfully loaded JSON file : {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON file : {path}")
        raise e


@typechecked
def save_json(path: Path | str, data: Any):
    import json

    try:
        with open(path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        logger.info(f"Successfully saved JSON file : {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON file : {path}")
        raise e


@typechecked
def create_directories(path: Path | str):
    if isinstance(path, str):
        path = Path(path)
    try:
        if not path.exists():
            os.makedirs(path, exist_ok=True)
            logger.info(f"Successfully created directory : {path}")
        else:
            logger.info(f"Directory Already Exists : {path}")
    except Exception as e:
        logger.error(f"Failed to create Directory : {path}")
        raise e
