import argparse
import time
import signal
import os
import cv2
import torch
import numpy as np
from models.experimental_yolo import attempt_load
from utils_yolo.datasets import LoadRs, LoadImages
from utils_yolo.general import check_img_size, set_logging
from utils_yolo.torch_utils import select_device, time_synchronized
from utils_yolo.general import non_max_suppression_kpt, output_to_keypoint
from utils.skeleton import Skeleton3D


def detect(weights, view_img, imgsz, annot, source, subsampling=0):
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    if half:
        model.half()  # to FP16
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    if source == "rs":
        dataset = LoadRs(img_size=imgsz, stride=stride)
    else:
        assert os.path.exists(source), "Check the source folder of the dataset!"
        dataset = LoadImages(source)
    if annot:
        annot_path = os.path.join(source, opt.annot_file_name)
        assert annot_path.endswith(".json"), "Annotations file must be a json file"
        sk = Skeleton3D(annot=annot, show_images=view_img, annot_name=annot_path)
    else:
        # Object to manage the 3D skeleton
        sk = Skeleton3D(annot=annot, show_images=view_img)

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    t0 = time.time()

    # To control termination
    def termination_handler(signum, frame):
        print(f"Done. ({time.time() - t0:.3f}s)")
        if annot:
            sk.write_annotations(annot_path)
        exit(1)

    signal.signal(signal.SIGINT, termination_handler)

    skip = 0  # The number of in-between frames to skip is subsampling
    for img, depth, _ in dataset:
        if skip < subsampling:
            skip += 1
            continue
        else:
            skip = 0
        if img is None:
            break
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != "cpu" and (
            old_img_b != img.shape[0]
            or old_img_h != img.shape[2]
            or old_img_w != img.shape[3]
        ):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for _ in range(3):
                model(img, augment=False)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression_kpt(
            pred,
            0.25,
            0.65,
            nc=model.yaml["nc"],
            nkpt=model.yaml["nkpt"],
            kpt_label=True,
        )
        t3 = time_synchronized()

        # Process detections
        with torch.no_grad():
            pred = output_to_keypoint(pred)
        nimg = img[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

        for idx in range(pred.shape[0]):
            sk.plot_skeleton(nimg, depth=depth, kpts=pred[idx, 7:].T, steps=3)

            # Print time (inference + NMS)
            print(
                f"Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS"
            )
            break  # Just take into account the first detected skeleton

    print(f"Done. ({time.time() - t0:.3f}s)")
    if annot:
        sk.write_annotations(annot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default="weights/yolov7-w6-pose.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--source", type=str, default="rs", help="Source"
    )  # file/folder, rs for realsense
    parser.add_argument(
        "--img-size", type=int, default=640, help="Inference size (pixels)"
    )
    parser.add_argument(
        "--device", default="", help="Cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="Display results")
    parser.add_argument(
        "--annot", action="store_true", help="Write skeleton annotations"
    )
    parser.add_argument(
        "--annot-file-name",
        default="annotations_3fps.json",
        help="Annotations file name with .json extension",
    )
    opt = parser.parse_args()
    print(opt)
    weights, view_img, imgsz, annot, source = (
        opt.weights,
        opt.view_img,
        opt.img_size,
        opt.annot,
        opt.source,
    )
    with torch.no_grad():
        if source == "rs":
            detect(weights, view_img, imgsz, annot, source)
        else:
            folders = source.split("/")
            if folders[-1] == "":
                folders.pop()
            content = os.listdir(source)
            is_gesture_folder = np.any([i == "color" for i in content]) and np.any(
                [i == "depth" for i in content]
            )
            if is_gesture_folder:  # Folder containing one gesture record
                c = folders[-2]
                print(
                    "Writing skeleton for "
                    + c
                    + " gesture, example "
                    + str(int(folders[-1]))
                )
                detect(weights, view_img, imgsz, annot, os.path.join(source))
            else:
                nexts = os.listdir(source)
                nexts.sort()
                content = os.listdir(os.path.join(source, nexts[0]))
                is_gesture_folder = np.any([i == "color" for i in content]) and np.any(
                    [i == "depth" for i in content]
                )
                if is_gesture_folder:  # Folder containing a single gesture records
                    c = folders[-1]
                    for n in nexts:
                        print(
                            "Writing skeleton for "
                            + folders[-1]
                            + " gesture, example "
                            + str(int(n))
                        )
                        detect(weights, view_img, imgsz, annot, os.path.join(source, n))
                else:  # Folder containing many gestures
                    i = 0
                    for c in nexts:
                        then = os.listdir(os.path.join(source, c))
                        then.sort()
                        content = os.listdir(os.path.join(source, c, then[0]))
                        is_gesture_folder = np.any(
                            [i == "color" for i in content]
                        ) and np.any([i == "depth" for i in content])
                        if not is_gesture_folder:
                            continue
                        records = os.listdir(os.path.join(source, c))
                        records.sort()
                        for record in records:
                            print(
                                "Writing skeleton for "
                                + c
                                + " gesture, example "
                                + str(int(record))
                            )
                            detect(
                                weights,
                                view_img,
                                imgsz,
                                annot,
                                os.path.join(source, c, record),
                            )
