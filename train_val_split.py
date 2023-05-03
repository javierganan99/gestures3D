import argparse
import os
import shutil
import math
import random
from utils.auxiliary import ensure_exist


def main(opt, extension=".json"):
    classes = os.listdir(opt.dataset_path)
    assert os.path.exists(opt.dataset_path), "Dataset not found in " + opt.dataset_path + " path."
    ensure_exist(os.path.join(opt.output_folder, "val"))
    ensure_exist(os.path.join(opt.output_folder, "train"))
    cont_train = 0
    cont_val = 0
    for c in classes:
        class_folder = os.path.join(opt.dataset_path, c)
        files = [f for f in os.listdir(class_folder) if f.endswith(extension)]
        val_number = math.ceil(opt.val_size * len(files))
        val_files = set(random.sample(files, val_number))
        train_files = set(files) - val_files
        for vf in val_files:
            shutil.copy(
                os.path.join(class_folder, vf),
                os.path.join(opt.output_folder, "val", str(cont_val).zfill(7) + extension),
            )
            cont_val += 1
        for tf in train_files:
            shutil.copy(
                os.path.join(class_folder, tf),
                os.path.join(opt.output_folder, "train", str(cont_train).zfill(7) + extension),
            )
            cont_train += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="dataset",
        help="Input dataset with data separated into folder corresponding to classes",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Folder in which to output the dataset splitted into train and val folders",
    )
    parser.add_argument("--val_size", type=float, default=0.15, help="validation set size")
    opt = parser.parse_args()
    print(opt)
    main(opt)
