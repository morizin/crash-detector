import torch
import torch.nn as nn
import torchvision.models as models
from typeguard import typechecked
from .loss import get_loss_function
from typing import Collection
from ...config.config_entity import ModelTrainingConfig


class Model(nn.Module):
    @typechecked
    def __init__(self, config: ModelTrainingConfig):
        super().__init__()

        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        self.criterion = get_loss_function(self.config.loss)

    @typechecked
    def forward(self, inputs: Collection[torch.Tensor | None]) -> torch.Tensor:
        return self.backbone(inputs)

    @typechecked
    def forward_step(self, inputs: Collection[torch.Tensor | None]) -> torch.Tensor:
        images, labels = inputs
        images = (
            images.to(self.device, non_blocking=True).float().permute(0, 3, 1, 2)
            / 255.0
        )
        logits = self(images)

        if labels is not None:
            labels = labels.to(self.device, non_blocking=True).float().unsqueeze(1)
            loss = self.criterion(logits, labels)
            return loss
        else:
            return logits
