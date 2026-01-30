import torch
from ...config.config_entity import ModelExportingConfig
from ...config.artifact_entity import ModelExportingArtifact
from .model import Model
import logging

logger = logging.getLogger(__name__)


class ModelExportingComponent:
    def __init__(self, config: ModelExportingConfig):
        self.config = config
        self.model = Model.from_pretrained(model_path=config.model_path)
        self.model.eval()
        self.export_path = config.outdir

    def export_to_onnx(
        self,
    ):
        dummy_input = torch.randn(2, 20, 224, 224).to(self.model.device)
        return torch.onnx.export(
            model=self.model,
            f=self.export_path / "model.onnx",
            args=dummy_input,
            opset_version=self.config.onnx_opset,
            input_names=["inputs"],
            dynamic_shapes={"inputs": {0: "batch_size"}},
            external_data=False,
            dynamo=True,
            artifacts_dir=str(self.export_path),
            report=True,
            verify=True,
            # profile=True
        )

    def __call__(
        self,
    ):
        _ = self.export_to_onnx()
        logger.info(f"ONNX model exported at : {self.export_path / 'model.onnx'}")

        return ModelExportingArtifact(
            name=self.config.name,
            model_path=self.config.model_path,
            onnx_model_path=self.export_path / "model.onnx",
            input_shape=None,
            output_shape=None,
        )
