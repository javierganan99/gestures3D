import os
import numpy as np
import argparse
import sys
import pickle
from numpy.lib.format import open_memmap
from utils.skeleton import Skeleton3D, SkeletonFeeder
from utils.transformations import translate_to_center_batch, random_rotation
from utils.auxiliary import ensure_exist

toolbar_width = 30


def print_toolbar(rate, annotation=""):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(" ")
        else:
            sys.stdout.write("-")
        sys.stdout.flush()
    sys.stdout.write("]\r")


def end_toolbar():
    sys.stdout.write("\n")


def main(arg):
    folders = os.listdir(arg.dataset_path)
    ensure_exist(arg.output_folder)
    for folder in folders:
        feeder = SkeletonFeeder(  # create an object to load the data
            data_path=os.path.join(arg.dataset_path, folder),
            num_person_in=1,
            num_person_out=1,
        )
        print("Currently showing " + folder + " gestures...")
        fp = open_memmap(  # Create a numpy object to save the data into numpy format
            os.path.join(arg.output_folder, folder + "_data.npy"),
            dtype="float32",
            mode="w+",
            shape=(len(feeder.sample_name), 4, 120, 17, 1),
        )
        print("Number of gestures for {}: {}".format(folder, len(feeder.sample_name)))
        sk = Skeleton3D()  # create an object to plot the skeleton
        sample_label = []
        for i, _ in enumerate(feeder.sample_name):
            data, label = feeder[i]
            print_toolbar(
                i * 1.0 / len(feeder.sample_name),
                "({:>5}/{:<5}) Processing data: ".format(i + 1, len(feeder.sample_name)),
            )
            # Custom transformations
            data = translate_to_center_batch(data)
            data = random_rotation(data)
            fp[i, :, 0 : data.shape[1], :, :] = data
            sample_label.append(label)
            with open(os.path.join(arg.output_folder, folder + "_label.pkl"), "wb") as f:
                pickle.dump((feeder.sample_name, list(sample_label)), f)

            if arg.show:
                for j in range(data.shape[1]):
                    skl = data[:-1, j, :, :]
                    skl = np.ravel(skl, order="F")
                    conf = data[-1, j, :, :]
                    conf = np.ravel(conf, order="F")
                    sk.plot_3D_skeleton(skl, conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skeleton Data Converter.")
    parser.add_argument("--dataset-path", type=str, default="dataset", help="Path of the dataset to be converted")
    parser.add_argument(
        "--output-folder", type=str, default="dataset_numpy", help="Path of the numpy converted dataset"
    )
    parser.add_argument(
        "--show", action="store_true", help="Wheter to show the skeleton. It significantly slows down the conversion."
    )
    opt = parser.parse_args()
    print(opt)
    main(opt)
