from ..core.io_types import Directory
from pydantic import BaseModel, field_validator
from pathlib import Path
from typing import Optional
from .. import logger
from ..utils.common import load_yaml
from ..constants import SCHEMA_DIR
import os


class DataSchema(BaseModel):
    name: str
    path: Path | str | Directory

    train: Optional[list[str | Path]] = None
    train_image_folder: Optional[str | Path] = None
    valid: Optional[list[str | Path]] = None
    valid_image_folder: Optional[str | Path] = None
    test: Optional[list[str | Path]] = None
    test_image_folder: Optional[str | Path] = None

    columns: Optional[dict[str, str]] = None
    categorical: Optional[list[str]] = None
    target: Optional[str] = None
    additional_properties: Optional[dict[str, str]] = None

    def model_post_init(self, __context__):
        file_path = os.path.join(SCHEMA_DIR, f"{self.name}.yaml")
        if os.path.exists(file_path):
            content = load_yaml(file_path)

            self.path = Directory(path=self.path)

            self.train = [self.path / f for f in content["train"]]
            self.train_image_folder = Path(content["train_image_folder"])

            self.valid = (
                [self.path / f for f in content["valid"]]
                if hasattr(content, "valid")
                else None
            )
            self.valid_image_folder = (
                Path(content["valid_image_folder"])
                if hasattr(content, "valid_image_folder")
                else None
            )

            self.test = [self.path / f for f in content["test"]]
            self.test_image_folder = Path(content["test_image_folder"])

            self.columns = content["columns"]
            self.categorical = content.get("categorical", [])
            self.additional_properties = content.get("additional-properties", {})
            self.target = content["target"]
            return self
        else:
            e = FileNotFoundError(f"Schema file {self.name} does not exist.")
            logger.error(e)
            raise e


class DataValidataionConfig(BaseModel):
    indir: Directory
    outdir: Directory
    report_path: Path | str
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
    schemas: dict[str, DataSchema]
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

    source_model: str
    loss: str
    metrics: list[str]
    optimizer: str
    learning_rate: float
    train_batch_size: int
    valid_batch_size: int
    n_epochs: int
    gradient_accumulation_steps: int

    outdir: Directory


class ModelEvaluationConfig(BaseModel):
    name: str
    valid_file_path: Path
    test_file_path: Path
    metrics: list[str]
    model_path: Path | str
