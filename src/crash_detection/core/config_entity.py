from .io_types import Directory
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
from .. import logger
from ..utils.common import load_yaml
from ..constants import SCHEMA_DIR
import os


class DataSchema(BaseModel):
    name: str

    train: Optional[str] = None
    train_image_folder: Optional[str] = None
    test: Optional[str] = None
    test_image_folder: Optional[str] = None

    columns: Optional[dict[str, str]] = None
    categorical: Optional[list[str]] = None
    target: Optional[str] = None
    additional_properties: Optional[dict[str, str]] = None

    def model_post_init(self, __context__):
        file_path = os.path.join(SCHEMA_DIR, f"{self.name}.yaml")
        if os.path.exists(file_path):
            content = load_yaml(file_path)

            self.train = content["train"]
            self.train_image_folder = content["train_image_folder"]
            self.test = content["test"]
            self.test_image_folder = content["test_image_folder"]

            self.columns = content["columns"]
            self.categorical = content.get("categorical", [])
            self.additional_properties = content.get("additional_properties", {})
            self.target = content["target"]
            return self
        else:
            e = FileNotFoundError(f"Schema file {self.name} does not exist.")
            logger.error(e)
            raise e


class DataValidataionConfig(BaseModel):
    report_name: Path | str
    indir: Directory
    outdir: Directory
    pixel_histogram: bool
    statistics: bool
    kl_divergence: bool
    schemas: dict[str, DataSchema]


class DataSource(BaseModel):
    id: str
    source: str
    type: Optional[str] = None
    link: str


class DataIngestionConfig(BaseModel):
    data_sources: dict[str, DataSource]
    outdir: Directory


class DataSplitConfig(BaseModel):
    type: str
    ratio: float = 0.2
    n_splits: int = 5

    def model_post_init(self, __context__):
        train_splits = 10
        while (train_splits * self.ratio) % 1 != 0:
            train_splits *= 10

        self.n_splits = train_splits

        if self.type not in ["random", "kfold", "skfold"]:
            e = ValueError(f"Data split type {self.type} is not supported.")
            logger.error(e)
            raise e


class DataTransformationConfig(BaseModel):
    indir: Directory
    datasets: dict[str, DataSchema]
    split: Optional[DataSplitConfig] = None
    frames_per_clip: int
    resize: dict[str, int]
    normalize: bool
    grayscale: bool
    outdir: Directory


class ModelTrainingConfig(BaseModel):
    name: str
    type: str
    transforms: Optional[DataTransformationConfig] = None
    outdir: Directory
