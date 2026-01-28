import torch
import torch.nn as nn
import torchvision.models as models
from typeguard import typechecked
from typing import Collection
from ...config.config_entity import ModelTrainingConfig
from ...utils.common import load_pickle
from pathlib import Path
from .loss import get_loss_function
import os


class Model(nn.Module):
    @typechecked
    def __init__(self, config: ModelTrainingConfig):
        super().__init__()

        self.config = config

        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if config.source_model == "resnet18":
            self.backbone = models.resnet18(weights=None)
        elif config.source_model == "resnet34":
            self.backbone = models.resnet34(weights=None)
        else:
            raise ValueError(f"Unsupported model source: {config.source_model}")

        self.backbone.conv1 = nn.Conv2d(
            20, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone.fc = nn.Linear(512, 1)

        self.criterion = get_loss_function(config.loss)

        if self.device != "cpu":
            self.to(self.device)

    @classmethod
    def from_pretrained(
        cls, model_path: Path | str, config_path: Path | str | None = None
    ) -> "Model":
        model_state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model = cls(
            config=ModelTrainingConfig(
                **load_pickle(
                    os.path.join(
                        os.path.dirname(model_path), "model_training_config.pkl"
                    )
                )
            )
        )
        model.load_state_dict(model_state_dict)
        model.to(model.device)
        return model

    @typechecked
    def forward(self, inputs: Collection[torch.Tensor | None]) -> torch.Tensor:
        return self.backbone(inputs)

    @typechecked
    def forward_step(
        self, inputs: Collection[torch.Tensor | None], get_logits: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        images, labels = inputs
        images = (
            images.to(self.device, non_blocking=True)
            .float()
            .permute(0, 3, 1, 2)
            .contiguous()
            / 255.0
        )
        logits = self(images)

        if labels is not None:
            labels = labels.to(self.device, non_blocking=True).float().unsqueeze(1)
            loss = self.criterion(logits, labels)
            if get_logits:
                return loss, logits
            else:
                return loss
        else:
            return logits
