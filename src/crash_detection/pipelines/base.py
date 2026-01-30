from ..core import Directory
from ..config import ConfigurationManager
from ..config.config_entity import (
    DataIngestionConfig,
    DataValidataionConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
    ModelExportingConfig,
)
from ..config.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainingArtifact,
    ModelEvaluationArtifact,
    ModelExportingArtifact,
)

from typeguard import typechecked

from ..components.data.ingestion import DataIngestionComponent
from ..components.data.transformation import DataTransformationComponent
from ..components.data.validation import DataValidationComponent
from ..components.model.train import ModelTrainingComponent
from ..components.model.eval import ModelEvaluationComponent
from ..components.model.export import ModelExportingComponent
from ..utils.common import save_pickle
import logging

logger = logging.getLogger(__name__)


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

            save_pickle(
                path=self.config.artifact_path
                // "config"
                / "data_ingestion_config.pkl",
                data=data_ingestion_config.model_dump(),
            )

            data_ingestion_artifact: DataIngestionArtifact = DataIngestionComponent(
                data_ingestion_config
            )()

            logger.info("Data Ingesion Completed")
            save_pickle(
                path=self.config.artifact_path
                // "artifacts"
                / "data_ingestion_artifact.pkl",
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

            save_pickle(
                path=self.config.artifact_path
                // "config"
                / "data_validation_config.pkl",
                data=data_validation_config.model_dump(),
            )

            data_validation_artifact = DataValidationComponent(
                config=data_validation_config,
            )()
            logger.info("Data Validation Completed")

            save_pickle(
                path=self.config.artifact_path
                // "artifacts"
                / "data_validation_artifact.pkl",
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

            save_pickle(
                path=model_path / "data_transformation_config.pkl",
                data=data_transformation_config.model_dump(),
            )

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
    def do_model_training(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_training_config: ModelTrainingConfig,
    ) -> ModelTrainingArtifact:
        try:
            save_pickle(
                path=model_training_config.outdir / "model_training_config.pkl",
                data=model_training_config.model_dump(),
            )

            model_trainer_artifact: ModelTrainingArtifact = ModelTrainingComponent(
                config=model_training_config,
                data_transformation_artifact=data_transformation_artifact,
            )()

            save_pickle(
                path=model_training_config.outdir / "model_training_artifact.pkl",
                data=model_trainer_artifact.model_dump(),
            )

            return model_trainer_artifact
        except Exception as e:
            logger.error(f"Error during model training {e}")
            raise e

    @typechecked
    def do_model_evaluation(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_training_artifact: ModelTrainingArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            logger.info("Model Evaluation ....")

            model_evaluation_config: ModelEvaluationConfig = (
                self.config.get_model_evaluation_config(
                    data_transformation_artifact=data_transformation_artifact,
                    model_training_artifact=model_training_artifact,
                )
            )

            save_pickle(
                path=model_training_artifact.outdir / "model_evaluation_config.pkl",
                data=model_evaluation_config.model_dump(),
            )

            model_evaluation_artifact = ModelEvaluationComponent(
                config=model_evaluation_config
            )()

            logger.info("Model Evaluation Completed")

            save_pickle(
                path=model_training_artifact.outdir / "model_evaluation_artifact.pkl",
                data=model_evaluation_artifact.model_dump(),
            )

            return model_evaluation_artifact
        except Exception as e:
            logger.error(f"Error during model evaluation {e}")
            raise e

    @typechecked
    def do_model_exporting(
        self, model_training_artifact: ModelTrainingArtifact
    ) -> ModelExportingArtifact:
        try:
            logger.info("Model Exporting ....")

            model_exporting_config: ModelExportingConfig = (
                self.config.get_model_exporting_config(
                    model_training_artifact=model_training_artifact
                )
            )

            save_pickle(
                path=model_training_artifact.outdir / "model_exporting_config.pkl",
                data=model_exporting_config.model_dump(),
            )

            model_exporting_artifact = ModelExportingComponent(
                config=model_exporting_config
            )()

            logger.info("Model Exporting Completed")
            save_pickle(
                path=model_training_artifact.outdir / "model_exporting_artifact.pkl",
                data=model_exporting_artifact.model_dump(),
            )
            return model_exporting_artifact

        except Exception as e:
            logger.error(f"Error during model exporting {e}")
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

        logger.info("Model Trainer ....")
        model_trainer_config: dict[str, ModelTrainingConfig] = (
            self.config.get_model_training_config(
                data_validation_artifact=data_validation_artifact
            )
        )

        model_trainer_artifacts: dict[str, ModelTrainingArtifact] = {}
        model_evaluation_artifacts: dict[str, ModelEvaluationArtifact] = {}
        model_exporting_artifacts: dict[str, ModelExportingArtifact] = {}
        for model, model_config in model_trainer_config.items():
            data_transformation_artifact: DataTransformationArtifact = (
                self.do_data_transformation(
                    model_config.transforms, model_path=model_config.outdir
                )
            )

            model_trainer_artifacts[model] = self.do_model_training(
                data_transformation_artifact=data_transformation_artifact,
                model_training_config=model_config,
            )

            model_evaluation_artifacts[model] = self.do_model_evaluation(
                data_transformation_artifact=data_transformation_artifact,
                model_training_artifact=model_trainer_artifacts[model],
            )

            model_exporting_artifacts[model] = self.do_model_exporting(
                model_training_artifact=model_trainer_artifacts[model]
            )

        logger.info("Base Pipeline Completed")

        print("Model Training Artifacts:")
        print(model_trainer_artifacts)
        print("Model Evaluation Artifacts:")
        print(model_evaluation_artifacts)
        print("Model Exporting Artifacts:")
        print(model_exporting_artifacts)

        return (
            model_trainer_artifacts,
            model_evaluation_artifacts,
            model_exporting_artifacts,
        )
