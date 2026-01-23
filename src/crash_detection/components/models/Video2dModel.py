import torch.nn as nn
import torchvision.models as models


class Video2dModel(nn.Module):
    def __init__(self, model_source: str = "resnet18"):
        super().__init__()
        if model_source == "resnet18":
            self.backbone = models.resnet18(weights=None)
        elif model_source == "resnet34":
            self.backbone = models.resnet34(weights=None)
        else:
            raise ValueError(f"Unsupported model source: {model_source}")

        self.backbone.conv1 = nn.Conv2d(
            20, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone.fc = nn.Linear(512, 1)

    def forward(self, inputs):
        return self.backbone(inputs)
