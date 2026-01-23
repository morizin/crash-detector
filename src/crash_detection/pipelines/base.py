from ..core import Directory
from ..config import ConfigurationManager
from ..config.config_entity import (
    DataIngestionConfig,
    DataValidataionConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
)
from ..config.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainingArtifact,
)

from typeguard import typechecked
from .. import logger

from ..components.data.ingestion import DataIngestionComponent
from ..components.data.transformation import DataTransformationComponent
from ..components.data.validation import DataValidationComponent
from ..components.training import ModelTrainingComponent
from ..utils.common import save_pickle


class BasePipeline:
    def __init__(self):
        try:
            logger.info("Configuring....")
            self.config: ConfigurationManager = ConfigurationManager()
        except Exception as e:
            logger.error(f"Error Configuring... {e}")

    @typechecked
    def do_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Data Ingestion ....")
            data_ingestion_config: DataIngestionConfig = (
                self.config.get_data_ingestion_config()
            )

            data_ingestion_artifact: DataIngestionArtifact = DataIngestionComponent(
                data_ingestion_config
            )()

            logger.info("Data Ingesion Completed")
            save_pickle(
                path=self.config.artifact_path / "data_ingestion_artifact.pkl",
                data=data_ingestion_artifact.model_dump(),
            )
            return data_ingestion_artifact

        except Exception as e:
            logger.error(f"Error during data ingestion {e}")
            raise e

    @typechecked
    def do_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
    ) -> DataValidationArtifact:
        try:
            logger.info("Data Validation ....")
            data_validation_config: DataValidataionConfig = (
                self.config.get_data_validation_config(
                    data_ingestion_artifact=data_ingestion_artifact
                )
            )

            data_validation_artifact = DataValidationComponent(
                config=data_validation_config,
            )()
            logger.info("Data Validation Completed")

            save_pickle(
                path=self.config.artifact_path / "data_validation_artifact.pkl",
                data=data_validation_artifact.model_dump(),
            )
            return data_validation_artifact
        except Exception as e:
            logger.error(f"Error during data validation {e}")
            raise e

    @typechecked
    def do_data_transformation(
        self,
        data_transformation_config: DataTransformationConfig,
        model_path: Directory,
    ) -> DataTransformationArtifact:
        try:
            logger.info("Data Transformation ....")
            data_transformation_artifact = DataTransformationComponent(
                config=data_transformation_config
            )()
            logger.info("Data Transformation Completed")
            save_pickle(
                path=model_path / "data_transformation_artifact.pkl",
                data=data_transformation_artifact.model_dump(),
            )
            return data_transformation_artifact
        except Exception as e:
            logger.error(f"Error during data transformation {e}")
            raise e

    @typechecked
    def do_model_trainer(
        self,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            logger.info("Model Trainer ....")
            model_trainer_config: dict[str, ModelTrainingConfig] = (
                self.config.get_model_training_config(
                    data_validation_artifact=data_validation_artifact
                )
            )

            model_trainer_components: dict[str, ModelTrainingArtifact] = {}
            for model, config in model_trainer_config.items():
                data_transformation_artifact: DataTransformationArtifact = (
                    self.do_data_transformation(
                        config.transforms, model_path=config.outdir
                    )
                )

                model_trainer_components[model] = ModelTrainingComponent(
                    config=config,
                    data_transformation_artifact=data_transformation_artifact,
                )()
            logger.info("Model Trainer Completed")
            return model_trainer_components
        except Exception as e:
            logger.error(f"Error during model trainer {e}")
            raise e

    @typechecked
    def kickoff(
        self,
    ):
        logger.info("Kicking off Base Pipeline")
        data_ingestion_artifact: DataIngestionArtifact = self.do_data_ingestion()

        data_validation_artifact: DataValidationArtifact = self.do_data_validation(
            data_ingestion_artifact=data_ingestion_artifact
        )

        model_trainer_components = self.do_model_trainer(
            data_validation_artifact=data_validation_artifact
        )
        logger.info("Base Pipeline Completed")

        return model_trainer_components
