import subprocess
import gdown
import os


GTA_CRASH_IMAGES = [
    "1xTqFxguYxvF8zf7_e_clAq2GTq1Wi3vC",
    "1K_wEYFvyqMI_Dq_Au8d97Fb23RgCzS40",
    "1JjuJ-h670FWYsaZ5V7XMGPNg8QNertRU",
]

YT_CRASH_IMAGES = ["1mC1nlkLa6ffU09ChbQLK7d7sN_B-rMxP"]

GTA_SAFE_IMAGES = [
    "1fSEqEvhDm-vKm4ZPSddn08w6NuV3Zm4l",
    "1Q8xQMWrdbzSTjCa8Cr68Cnx1gN5DxTfP",
]

YT_SAFE_IMAGES = ["1Ofpe5ZKlf3pWix8Ho3Zm-7ZpxEIH_8qL"]

TEST_CRASH_IMAGES = [
    "1xGn2O-N6_ONAwWnbjQKSpcTloU6QLapR",
]

TEST_SAFE_IMAGES = ["15VMXowVcbZVVm7aOw2wsgCuzqmUMZanX"]

GTA_CRASH_LABELS = [
    "1LKHVBPeadPzbMZjsXyALD0ERMyayq1vv",
    "1CJmC21G4UOM1B2WnMax_aIkvk9X-Ho-R",
    "1Ikgkpl4EbZga2IrymInWlzyUmuWNK9eB",
]

YT_CRASH_LABELS = ["1lkeswrHasRF1tH6selq5kp-JTbCvfkjS"]

GTA_SAFE_LABELS = [
    "13G_MUZ00dw12YhJE-oxCI0KeH8UGzoWR",
    "1zNs9YUDzXzYLuUunTCmQQPCHm51qjHHv",
]

YT_SAFE_LABELS = ["1q-VaEHBgSCb_ugob7TPVZrqQX_9CtdUc"]

TEST_CRASH_LABELS = [
    "11fdzKTSensNlwQpau0Gi8Ikraf0yhYcQ",
]

TEST_SAFE_LABELS = ["1lLEY3Gj6sWJCDcGLi0RYomeWCtTHXQIt"]

for name in ["GTA", "TEST", "YT"][:2]:
    for label in ["CRASH", "SAFE"]:
        src_folder = f"/kaggle/working/{name}_{label}"
        for images, labels in zip(
            eval(f"{name}_TRAIN_{label}_IMAGES"), eval(f"{name}_TRAIN_{label}_LABELS")
        ):
            print(f"{images = } {labels = }")
            gdown.download(id=images, resume=True)
            subprocess.call(
                f"tar -xf {os.path.join(src_folder, 'images.tar.gz')}".split()
            )
            subprocess.call(
                f"rm -f".split() + [os.path.join(src_folder, "images.tar.gz")]
            )

            gdown.download(id=labels, resume=True)
            subprocess.call(
                f"tar -xf {os.path.join(src_folder, 'labels.tar.gz')}".split()
            )
            subprocess.call(
                f"rm -f".split() + [os.path.join(src_folder, "labels.tar.gz")]
            )
