from pydantic import BaseModel
from ..core.io_types import Directory
from .config_entity import DataSchema
from pathlib import Path


class DataIngestionArtifact(BaseModel):
    schemas: dict[str, DataSchema]
    path: Directory


class DataValidationArtifact(BaseModel):
    report_path: Path
    image_dir: Directory
    valid_data_dir: Directory
    invalid_data_dir: Directory
    is_validated: bool
    schemas: dict[str, DataSchema]


class DataTransformationArtifact(BaseModel):
    path: Directory
    train_file_path: Path
    valid_file_path: Path
    test_file_path: Path
    schemas: dict[str, DataSchema]
