import onnxruntime as ort
import os
import numpy as np


def init():
    artifact_dir = os.path.join(
        "./artifacts", sorted([dt for dt in os.listdir("./artifacts/")])[-1], "onnx"
    )
    engines = [
        ort.InferenceSession(os.path.join(artifact_dir, model, "model.onnx"))
        for model in os.listdir(artifact_dir)
    ]
    return {
        "name": "my-stage",
        "engines": engines,
        "cache": np.zeros((20, 224, 224), dtype=np.float32),
        "count": 0,
    }
