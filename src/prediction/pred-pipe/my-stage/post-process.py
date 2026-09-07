import numpy as np


def hook(frame_data, context):
    preds = frame_data["inference_output"]
    preds = np.where(
        preds >= 0, 1 / (1 + np.exp(-preds)), np.exp(preds) / (1 + np.exp(preds))
    )
    frame_data["user_data"] = {"prediction": preds.tolist()}
