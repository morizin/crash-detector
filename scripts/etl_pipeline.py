import pandas as pd
import json
from tqdm.contrib.concurrent import thread_map
import numpy as np
import shutil
import os
from functools import partial
from tqdm.contrib.concurrent import thread_map

FINAL_DIR = "/kaggle/working"

NFRAMES = 20
DEBUG = -1

columns = [
    "id",
    "vid",
    "fid",
    "filename",
    "acceleration",
    "speed",
    "angularVelocity_x",
    "angularVelocity_y",
    "angularVelocity_z",
    "forwardV_x",
    "forwardV_y",
    "forwardV_z",
    "objectSize_x",
    "objectSize_y",
    "objectSize_z",
    "position_x",
    "position_y",
    "position_z",
    "vehicleCount",
    "vehicleInfo",
    "target",
]

GTA_TRAIN_CRASH_IMAGES = [
    "/kaggle/input/crash-dataset/GTACrash_accident_images_part1/GTACrash_accident_part1",
    "/kaggle/input/crash-dataset/GTACrash_accident_images_part2/GTACrash_accident_part2",
    "/kaggle/input/crash-dataset/GTACrash_accident_images_part3/GTACrash_accident_part3",
]

GTA_TRAIN_CRASH_LABELS = [
    "/kaggle/input/crash-dataset/GTACrash_accident_labels_part1/GTACrash_accident_part1",
    "/kaggle/input/crash-dataset/GTACrash_accident_labels_part2/GTACrash_accident_part2",
    "/kaggle/input/crash-dataset/GTACrash_accident_labels_part3/GTACrash_accident_part3",
]

GTA_TRAIN_SAFE_IMAGES = [
    "/kaggle/input/crash-dataset/GTACrash_nonaccident_images_part1/GTACrash_nonaccident_part1",
    "/kaggle/input/crash-dataset/GTACrash_nonaccident_images_part2/GTACrash_nonaccident_part2",
]

GTA_TRAIN_SAFE_LABELS = [
    "/kaggle/input/crash-dataset/GTACrash_nonaccident_labels_part1/GTACrash_nonaccident_part1",
    "/kaggle/input/crash-dataset/GTACrash_nonaccident_labels_part2/GTACrash_nonaccident_part2",
]


TEST_CRASH_IMAGES = [
    "/kaggle/input/crash-dataset/YouTubeCrash_test_accident_images/YouTubeCrash_test_accident"
]

TEST_CRASH_LABELS = [
    "/kaggle/input/crash-dataset/YouTubeCrash_test_accident_labels/YouTubeCrash_test_accident"
]

TEST_SAFE_IMAGES = [
    "/kaggle/input/crash-dataset/YouTubeCrash_test_nonaccident_images/YouTubeCrash_test_nonaccident"
]

TEST_SAFE_LABELS = [
    "/kaggle/input/crash-dataset/YouTubeCrash_test_nonaccident_labels/YouTubeCrash_test_nonaccident"
]


YT_TRAIN_CRASH_IMAGES = [
    "/kaggle/input/crash-dataset/YouTubeCrash_train_accident_images/YouTubeCrash_train_accident"
]

YT_TRAIN_CRASH_LABELS = [
    "/kaggle/input/crash-dataset/YouTubeCrash_train_accident_labels/YouTubeCrash_train_accident"
]

YT_TRAIN_SAFE_IMAGES = [
    "/kaggle/input/crash-dataset/YouTubeCrash_train_nonaccident_images/YouTubeCrash_train_nonaccident"
]

YT_TRAIN_SAFE_LABELS = [
    "/kaggle/input/crash-dataset/YouTubeCrash_train_nonaccident_labels/YouTubeCrash_train_nonaccident"
]


def load_json(path):
    annot = json.load(open(path, "r"))
    annot["id"] = os.path.basename(path)
    return annot


def label2csv(path, offset=0):
    records = thread_map(
        load_json,
        [
            os.path.join(path, file)
            for file in sorted(os.listdir(path))[
                : DEBUG if DEBUG > 0 else len(os.listdir(images))
            ]
        ],
        max_workers=4,
    )

    dataset = pd.DataFrame.from_records(records)

    composite_features = ["angularVelocity", "forwardV", "objectSize", "position"]
    try:
        for composite_feature in composite_features:
            dataset[[f"{composite_feature}_{i}" for i in "xyz"]] = dataset[
                composite_feature
            ].apply(pd.Series)
            dataset = dataset.drop([composite_feature], axis=1, inplace=False)
    except:
        pass

    dataset = dataset.sort_values(by="id").reset_index(drop=True)
    dataset["vid"] = np.repeat(
        offset + np.arange(dataset.shape[0] // NFRAMES + 1), NFRAMES
    )[: dataset.shape[0]]
    dataset["fid"] = np.repeat(
        np.arange(NFRAMES).reshape(1, NFRAMES),
        (dataset.shape[0] // NFRAMES + 1),
        axis=0,
    ).reshape(-1)[: dataset.shape[0]]
    dataset["filename"] = dataset.apply(
        lambda x: str(x["vid"]).zfill(5) + "_" + str(x["fid"]).zfill(2) + ".jpg", axis=1
    )

    return dataset, offset + dataset.shape[0] // NFRAMES


def process_img(dataset: pd.DataFrame, folder: str, dst_folder: str, idx: int) -> None:
    try:
        file_id = dataset.loc[idx, "id"].replace("json", "jpg")
        shutil.copy(
            os.path.join(folder, file_id),
            os.path.join(dst_folder, dataset.loc[idx, "filename"]),
        )
    except:
        print(file_id)
        raise


offset = 0
name = "GTA"
dst_folder = os.path.join(FINAL_DIR, "train_images")
os.makedirs(dst_folder, exist_ok=True)

datasets = []
for label in ["CRASH", "SAFE"]:
    temp = []
    for images, labels in zip(
        eval(f"{name}_TRAIN_{label}_IMAGES"), eval(f"{name}_TRAIN_{label}_LABELS")
    ):
        dataset, offset = label2csv(labels, offset)
        print(offset)
        p_process_img = partial(process_img, dataset, images, dst_folder)
        thread_map(p_process_img, range(len(dataset)), max_workers=4)

        if label == "CRASH":
            dataset["target"] = 1
        else:
            dataset["target"] = 0

        temp.append(dataset)
    datasets.append(pd.concat(temp, axis=0).reset_index(drop=True))

offset = 0
data = pd.concat(datasets, axis=0).reset_index(drop=True)
if all([col in data.columns for col in columns]):
    data = data[columns]

data.to_csv(os.path.join(os.path.dirname(dst_folder), "train.csv"), index=False)

offset = 0
name = "test"
dst_folder = os.path.join(FINAL_DIR, "test_images")
os.makedirs(dst_folder, exist_ok=True)

datasets = []
for label in ["CRASH", "SAFE"]:
    temp = []
    for images, labels in zip(
        eval(f"TEST_{label}_IMAGES"), eval(f"TEST_{label}_LABELS")
    ):
        dataset, offset = label2csv(labels, offset)
        print(offset)
        p_process_img = partial(process_img, dataset, images, dst_folder)
        thread_map(p_process_img, range(len(dataset)), max_workers=4)

        if label == "CRASH":
            dataset["target"] = 1
        else:
            dataset["target"] = 0

        temp.append(dataset)
    datasets.append(pd.concat(temp, axis=0).reset_index(drop=True))

offset = 0
data = pd.concat(datasets, axis=0).reset_index(drop=True)

data.to_csv(os.path.join(os.path.dirname(dst_folder), "test.csv"), index=False)

# offset = 0
# name = "YT"
# dst_folder = os.path.join(dst_folder, "train_images")
# os.makedirs(dst_folder, exist_ok = True)

# datasets = []
# for label in ["CRASH", "SAFE"]:
#     temp = []
#     for images, labels in zip(
#         eval(f"{name}_TRAIN_{label}_IMAGES"),
#         eval(f"{name}_TRAIN_{label}_LABELS")
#     ):
#         dataset, offset = label2csv(labels, offset)
#         print(offset)
#         p_process_img = partial(process_img, dataset, images, dst_folder)
#         thread_map(p_process_img, range(len(dataset)), max_workers=4)

#         if label == "CRASH":
#             dataset['target'] = 1
#         else:
#             dataset['target'] = 0

#         temp.append(dataset)
#     datasets.append(pd.concat(temp, axis = 0).reset_index(drop = True))

# offset = 0
# data = pd.concat(datasets, axis = 0).reset_index(drop = True)

# data.to_csv(
#     os.path.join(
#         os.path.dirname(dst_folder),
#         "train.csv"
#     ),
#     index = False
# )
