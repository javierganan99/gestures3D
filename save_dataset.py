import argparse
from utils_yolo.datasets import CreateRsDataset
from utils.auxiliary import HandleKeys


def main(opt):
    hk = HandleKeys(
        main_class=CreateRsDataset, class_parameters={"dataset_path": opt.dataset_path, "gesture": opt.gesture}
    )
    hk()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="output", help="path to output the dataset")
    parser.add_argument("--gesture", type=str, required=True, help="gesture folder in which to save the recording")
    opt = parser.parse_args()
    print(opt)
    main(opt)
