import numpy as np
import cv2
import argparse
import os
import torch
import signal
import time
from models.st_gcn_aaai18 import *
from models.experimental_yolo import attempt_load
from utils.skeleton import Skeleton3D, SkeletonBuffer
from utils_yolo.general import check_img_size, non_max_suppression_kpt, output_to_keypoint
from utils_yolo.torch_utils import select_device
from utils_yolo.datasets import LoadImages, LoadRs
from utils.auxiliary import load_yaml, ensure_exist


class DetectSkeleton:
    def __init__(self, device, source, weights, output_path=None, view_img=True, imgsz=640):
        self.device = device
        self.half = self.device.type != "cpu"  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        if self.half:
            self.model.half()  # to FP16
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if source == "rs":
            self.dataset = LoadRs(img_size=imgsz, stride=stride)
        else:
            assert os.path.exists(source), "Check the source folder of the dataset!"
            self.dataset = LoadImages(source)

        # Object to manage the 3D skeleton
        self.sk = Skeleton3D(annot=False, show_images=view_img, output_path=output_path)
        # Run inference
        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters()))
            )  # run once

    def __iter__(self):
        return self

    def __next__(self):
        img, depth, _ = next(self.dataset)
        if img is None:
            return None
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != "cpu" and (
            old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]
        ):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for _ in range(3):
                self.model(img, augment=False)[0]

        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression_kpt(
            pred, 0.25, 0.65, nc=self.model.yaml["nc"], nkpt=self.model.yaml["nkpt"], kpt_label=True
        )

        # Process detections
        with torch.no_grad():
            pred = output_to_keypoint(pred)
        nimg = img[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

        for idx in range(pred.shape[0]):
            # # Just take into account the first detected skeleton
            return self.sk.plot_skeleton(nimg, depth=depth, kpts=pred[idx, 7:].T, steps=3)


def predict():
    # Get the models' paths
    models_paths = load_yaml("cfg/weights.yaml")
    sk_model = models_paths["skeleton"]
    gestures_model = models_paths["gesture"]
    device = select_device(opt.device)
    # Buffer to store the consecutive skeletons
    buffer = SkeletonBuffer()
    # ST-GCN model to predict the gestures
    graph_cfg = {"layout": "coco"}
    model = ST_GCN_18(graph_cfg=graph_cfg, in_channels=4, num_class=6, edge_importance_weighting=True, data_bn=True)
    weights = torch.load(gestures_model)
    model.load_state_dict(weights)
    model.to(device)
    model.train(False)
    classes_dict = load_yaml("cfg/classes.yaml")
    # To save the inference in a video
    if opt.save:
        if ensure_exist(opt.output_path):
            examples = [int(i) for i in os.listdir(opt.output_path)]
            if examples:
                new_example = max(examples) + 1
            else:
                new_example = 1
        else:
            new_example = 1
        output_path = os.path.join(opt.output_path, str(new_example).zfill(5))
        ensure_exist(output_path)
        # Object to predict the skeleton
        ds = DetectSkeleton(device=device, source=opt.source, weights=sk_model, output_path=output_path)
    else:
        ds = DetectSkeleton(device=device, source=opt.source, weights=sk_model)

    # To control termination
    def termination_handler(signum, frame):
        time.sleep(1)
        exit(1)

    signal.signal(signal.SIGINT, termination_handler)
    cont = 0
    gesture = "Detecting gesture..."
    ds.sk.gesture = gesture
    while True:
        # 1st STEP: Predict skeleton
        result = next(ds)
        if result is None:
            print("Done!")
            break
        points3d, confidences = result
        # 2nd STEP: Append prediction to buffer
        buffer.append(points3d, confidences)
        print("Number of skeletons processed: {}".format(cont))
        cont += 1
        # 3rd STEP: Predict skeleton
        t0 = time.time()
        outputs = model(torch.from_numpy(buffer.buffer))
        t1 = time.time()
        print("Inference time for the gesture: {:.2f} s.".format(t1 - t0))
        prediction = torch.argmax(outputs, dim=1)
        gesture = classes_dict[int(prediction[0])]
        # Indicate the current detected gesture
        if cont < 120:  # Wait 3 seconds at 30 fps to load the buffer
            pass
        else:
            ds.sk.gesture = gesture.replace("_", " ")
        print("Gesture: {}".format(ds.sk.gesture))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="rs", help="Source")  # "folder" or "rs" for realsense
    parser.add_argument("--save", action="store_true", help="To save a video of the inference")
    parser.add_argument("--device", default="cpu", help="Cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--output_path", type=str, default="runs", help="Output path")
    opt = parser.parse_args()
    print(opt)
    predict()
