from .Video2dModel import Video2dModel
from box import ConfigDict
import torch.nn as nn
from typeguard import typechecked


@typechecked
def get_model(config: ConfigDict) -> nn.Module:
    if config.model_type == "video2d":
        return Video2dModel(model_source=config.model_source)
    else:
        raise ValueError(f"Unsupported model name: {config.model_type}")
