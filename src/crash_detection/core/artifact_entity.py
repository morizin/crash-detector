from pydantic import BaseModel
from .io_types import Directory
from pathlib import Path


class DataIngestionArtifact(BaseModel):
    names: list[str]
    path: Directory


class DataTransformationArtifact(BaseModel):
    path: Directory
    train_file_path: Path
    valid_file_path: Path
    test_file_path: Path
