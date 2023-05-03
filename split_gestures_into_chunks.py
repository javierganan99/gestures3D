import argparse
import os
from utils.skeleton import split_skeleton_annotation


def main(opt):
    classes = os.listdir(opt.dataset_path)
    for c in classes:
        records = os.listdir(os.path.join(opt.dataset_path, c))
        for r in records:
            files = [f for f in os.listdir(os.path.join(opt.dataset_path, c, r)) if f.endswith(".json")]
            for f in files:
                split_skeleton_annotation(
                    os.path.join(opt.dataset_path, c, r, f), opt.output_folder, max_T=opt.max_frames, stride=opt.stride
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="output", help="Base path of the input dataset")
    parser.add_argument("--output-folder", type=str, required=True, help="Base path of the output dataset")
    parser.add_argument("--max-frames", type=int, default=120, help="Maximun number of frames")
    parser.add_argument("--stride", type=int, default=3, help="Stride between 2 samples of frames")
    opt = parser.parse_args()
    print(opt)
    main(opt)
