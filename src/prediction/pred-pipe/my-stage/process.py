import numpy as np
import logging


def hook(frame_data, context):
    preds = []
    for engine in context["engines"]:
        inputs = {
            engine.get_inputs()[0].name: np.expand_dims(
                frame_data["inference_input"], axis=0
            )
        }
        outputs = engine.run(None, inputs)[0].flatten()[0]
        preds.append(outputs)

    frame_data["inference_output"] = np.array(preds)
