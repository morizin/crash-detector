from typeguard import typechecked
import yaml
from pathlib import Path
from box import ConfigBox
from typing import Any
from .. import logger
import os


@typechecked
def load_yaml(path: Path | str, boxed=True) -> ConfigBox | dict[str | Any]:
    if isinstance(path, str):
        path = Path(path)
    try:
        config = yaml.safe_load(open(path, "r"))
        logger.info(f"Successfully Loaded YAML file : {path}")
        if boxed:
            config = ConfigBox(config)
        return config
    except Exception as e:
        logger.error(f"Failed to load YAML file : {path}")
        raise e


@typechecked
def create_directories(path: Path | str):
    if isinstance(path, str):
        path = Path(path)
    try:
        if path.exists():
            os.makedirs(path, exist_ok=True)
            logger.info(f"Successfully created directory : {path}")
        else:
            logger.info(f"Directory Already Exists : {path}")
    except Exception as e:
        logger.error(f"Failed to create Directory : {path}")
        raise e
