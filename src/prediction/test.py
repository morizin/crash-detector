import cv2
import numpy as np

data = "../../data/transformed/gta-crash/train_images/00000.npy"
frames = np.load(data)

# Use lower FPS for short sequences
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 3.0, (224, 224), isColor=False)


for i in range(50):
    for f_id in range(frames.shape[-1]):
        frame = frames[..., f_id]
        if frame.dtype != np.uint8:
            frame = (
                (frame * 255).astype(np.uint8)
                if frame.max() <= 1.0
                else frame.astype(np.uint8)
            )

        if len(frame.shape) == 3:
            frame = (
                frame.squeeze()
                if frame.shape[2] == 1
                else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            )

        out.write(frame)

out.release()
print("Video saved (20 frames at 5 FPS = 4 seconds)")
