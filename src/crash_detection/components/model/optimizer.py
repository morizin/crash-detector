import torch
from torch.optim import Adam, SGD, Optimizer
from typeguard import typechecked
from ...config.config_entity import ModelTrainingConfig


@typechecked
def get_optimizer(config: ModelTrainingConfig, model_parameters) -> Optimizer:
    if config.optimizer == "adam":
        return Adam(model_parameters, lr=config.learning_rate)
    elif config.optimizer == "sgd":
        return SGD(model_parameters, lr=config.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer}")
