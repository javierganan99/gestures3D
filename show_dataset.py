import cv2
import argparse
from utils_yolo.datasets import LoadImages


def main(opt):
    loader = LoadImages(opt.dataset_path, opt.gesture)
    for img0, depth_image in loader:
        if img0 is None:
            break
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("img_color", img0)
        cv2.imshow("img_depth", depth_image)
        cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="output", help="base path of the dataset")
    parser.add_argument("--gesture", type=str, required=True, help="gesture folder to load")
    opt = parser.parse_args()
    print(opt)
    main(opt)
