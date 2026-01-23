from .. import logger, TIMESTAMP
from pathlib import Path
from .config_entity import (
    DataSource,
    DataIngestionConfig,
    DataValidataionConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
    DataSplitConfig,
)

from .artifact_entity import DataIngestionArtifact, DataValidationArtifact

from ..utils.common import load_yaml, seed_everything
from ..constants import (
    CONFIG_PATH,
    DATA_DIRECTORY_NAME,
    MODELS_DIRECTORY_NAME,
    SCHEMA_DIR,
    REPORT_NAME,
    RAW_DATA_DIRECTORY_NAME,
    TRANSFORMED_DATA_DIRECTORY_NAME,
)
from ..core import Directory
from ..core.hw_info import get_hw_details
import os


class ConfigurationManager:
    def __init__(self, config_path=CONFIG_PATH, latest: bool = False) -> None:
        self.config = load_yaml(config_path)

        if latest:
            latest_artifact_path = max(
                [
                    d
                    for d in os.listdir(self.config.artifact_path)
                    if os.path.isdir(os.path.join(self.config.artifact_path, d))
                ],
                default=None,
            )
            if latest_artifact_path is None:
                raise ValueError(
                    f"No artifact directories found in {self.config.artifact_path}"
                )

            logger.info(f"Using latest artifact path: {latest_artifact_path}")
            self.artifact_path = (
                Directory(path=self.config.artifact_path) // latest_artifact_path
            )

        else:
            self.artifact_path = Directory(path=self.config.artifact_path) // TIMESTAMP

        self.data_dir = Directory(path=DATA_DIRECTORY_NAME)
        self.schema_path = Path(SCHEMA_DIR)

        n_procs, n_threads = get_hw_details()
        logger.info(f"Number of Processors: {n_procs}")
        logger.info(f"Number of Threads per Processor: {n_threads}")

        self.config["hardware"] = {
            "n_procs": n_procs,
            "n_threads": n_threads,
        }

        if not os.path.exists(self.schema_path):
            logger.warning(f"Schema directory {self.schema_path} does not exist.")

        seed_everything(self.config.seed)

    def get_data_sources(self) -> dict[str, DataSource]:
        data_sources_v = dict()
        for name, source in self.config.data.items():
            data_sources_v[name] = DataSource(
                id=name,
                source=source.source,
                type=source.type if hasattr(source, "type") else None,
                link=source.link,
            )
        return data_sources_v

    def get_data_ingestion_config(
        self,
    ) -> DataIngestionConfig:
        outdir = self.data_dir // RAW_DATA_DIRECTORY_NAME

        return DataIngestionConfig(data_sources=self.get_data_sources(), outdir=outdir)

    def get_data_validation_config(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidataionConfig:
        return DataValidataionConfig(
            report_path=self.artifact_path / f"{REPORT_NAME}.json",
            indir=Directory(path=DATA_DIRECTORY_NAME) // RAW_DATA_DIRECTORY_NAME,
            outdir=Directory(path=DATA_DIRECTORY_NAME),
            pixel_histogram=False,
            statistics=False,
            kl_divergence=False,
            schemas=data_ingestion_artifact.schemas,
        )

    def get_model_training_config(
        self, data_validation_artifact: DataValidationArtifact
    ) -> dict[str, ModelTrainingConfig]:
        models = self.config.models
        model_configs = {}

        for model, params in models.items():
            model_configs[model] = ModelTrainingConfig(
                name=model,
                type=params.type,
                transforms=DataTransformationConfig(
                    indir=data_validation_artifact.valid_data_dir,
                    schemas={
                        name: schema
                        for name, schema in data_validation_artifact.schemas.items()
                        if name in params.datasets
                    },
                    split=DataSplitConfig(
                        type=params.transforms.split.type,
                        ratio=params.transforms.split.ratio,
                    )
                    if hasattr(params, "transforms")
                    and hasattr(params.transforms, "split")
                    else None,
                    frames_per_clip=params.transforms.frames_per_clip
                    if hasattr(params.transforms, "frames-per-clip")
                    else None,
                    resize=params.transforms.resize,
                    normalize=params.transforms.normalize
                    if hasattr(params.transforms, "normalize")
                    else False,
                    grayscale=params.transforms.grayscale
                    if hasattr(params.transforms, "grayscale")
                    else False,
                    outdir=self.data_dir // TRANSFORMED_DATA_DIRECTORY_NAME,
                )
                if hasattr(params, "transforms")
                else None,
                source_model=params.source_model
                if hasattr(params, "source-model")
                else None,
                loss=params.loss,
                metrics=params.metrics if hasattr(params, "metrics") else [],
                optimizer=params.optimizer,
                learning_rate=params.learning_rate,
                train_batch_size=params.train_batch_size,
                valid_batch_size=params.valid_batch_size,
                n_epochs=params.n_epochs,
                gradient_accumulation_steps=params.gradient_accumulation_steps,
                outdir=self.artifact_path // MODELS_DIRECTORY_NAME // model,
            )

        return model_configs
