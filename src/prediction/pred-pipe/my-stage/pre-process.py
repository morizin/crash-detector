import cv2
import numpy as np


def hook(frame_data, context):
    frame = frame_data["modified"]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32)
    frame /= 255

    inference_input = np.zeros((20, 224, 224), dtype=np.float32)
    frame_pos = frame_data["frame_number"] % 20
    inference_input[: 19 - frame_pos, ...] = context["cache"][frame_pos + 1 :, ...]
    context["cache"][frame_data["frame_number"] % 20, ...] = frame
    inference_input[19 - frame_pos :, ...] = context["cache"][: frame_pos + 1, ...]

    frame_data["inference_input"] = inference_input
