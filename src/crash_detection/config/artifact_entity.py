from pydantic import BaseModel
from ..core.io_types import Directory
from .config_entity import DataSchema
from pathlib import Path
from typing import Optional


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


class ClassificationArtifact(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None


class ModelTrainingArtifact(BaseModel):
    model_name: str
    model_path: Path
    train_loss: float | None = None
    valid_loss: float | None = None
    classification_artifact: Optional[ClassificationArtifact] = None


class ModelEvaluationArtifact(BaseModel):
    classification_artifact: Optional[ClassificationArtifact] = None
