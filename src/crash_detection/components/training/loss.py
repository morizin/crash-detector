import torch.nn as nn


def get_loss_function(loss_name: str):
    loss_name = loss_name.lower()
    if loss_name == "binary-cross-entropy":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "mean-squared-error":
        return nn.MSELoss()
    elif loss_name == "cross-entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
