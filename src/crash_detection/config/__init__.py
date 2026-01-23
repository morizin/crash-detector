from .. import logger, TIMESTAMP
from pathlib import Path
from ..core.config_entity import (
    DataSource,
    DataIngestionConfig,
    DataSchema,
    DataValidataionConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
    DataSplitConfig,
)

from ..core.artifact_entity import DataIngestionArtifact

from ..utils.common import load_yaml, seed_everything
from ..constants import (
    CONFIG_PATH,
    DATA_DIRECTORY_NAME,
    MODELS_DIRECTORY_NAME,
    SCHEMA_DIR,
    REPORT_NAME,
    TRANSFORMED_DATA_DIRECTORY_NAME,
)
from ..core import Directory
import os


class ConfigurationManager:
    def __init__(self, config_path=CONFIG_PATH):
        self.config = load_yaml(config_path)
        self.artifact_path = Directory(path=self.config.artifact_path) // TIMESTAMP
        self.data_dir = Directory(path=DATA_DIRECTORY_NAME)
        self.schema_path = Path(SCHEMA_DIR)

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
        outdir = self.data_dir // "raw"

        return DataIngestionConfig(data_sources=self.get_data_sources(), outdir=outdir)

    def get_data_validation_config(self):
        data_schemes = {}
        for d_name in self.config.data_sources:
            data_schemes[d_name] = DataSchema(name=d_name)

        return DataValidataionConfig(
            report_name=self.artifact_path
            // DATA_DIRECTORY_NAME
            / f"{REPORT_NAME}.yaml",
            indir=DATA_DIRECTORY_NAME,
            outdir=self.artifact_path // REPORT_NAME,
            pixel_histogram=False,
            statistics=False,
            kl_divergence=False,
            schemas=data_schemes,
        )

    def model_training_config(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> dict[str, ModelTrainingConfig]:
        models = self.config.models
        model_configs = {}

        for model, params in models.items():
            model_configs[model] = ModelTrainingConfig(
                name=model,
                type=params.type,
                transforms=DataTransformationConfig(
                    indir=data_ingestion_artifact.path,
                    datasets={
                        d_name: DataSchema(name=d_name) for d_name in params.datasets
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
                outdir=self.artifact_path // MODELS_DIRECTORY_NAME // model,
            )

        return model_configs
