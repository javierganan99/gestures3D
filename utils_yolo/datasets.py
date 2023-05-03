# Dataset utils and dataloaders
import time
from threading import Thread

import cv2
import numpy as np
import pyrealsense2 as rs
import os

from utils.auxiliary import ensure_exist


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class LoadRs:  # Load frames for the realsense
    def __init__(self, img_size=640, stride=64, fps=30, depth=True):
        self.depth = depth  # Return or not depth for the realsense
        self.mode = "stream"
        self.init_pipeline(fps)
        self.img_size = img_size
        self.stride = stride
        self.imgs = [None]
        self.depths = [None]
        # Check first image
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        self.imgs[0] = np.asanyarray(color_frame.get_data())
        self.depths[0] = depth_frame
        if depth:
            thread = Thread(target=self.update_rgbd, daemon=True)
        else:
            thread = Thread(target=self.update, daemon=True)
        thread.start()
        print("")  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print(
                "WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams."
            )

    def init_pipeline(self, fps):
        self.fps = fps
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                found_rgb = True
                break

        assert found_rgb, "Depth camera with Color sensor not found"

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.fps)

        if device_product_line == "L500":
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, self.fps)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.fps)

        # Start streaming
        profile = self.pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Align object
        self.align = rs.align(rs.stream.color)

    def update(self):
        # Read next stream frame in a daemon thread
        while True:
            # Get color_frame
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            # Convert image to numpy array
            self.imgs[0] = np.asanyarray(color_frame.get_data())
            time.sleep(1 / self.fps)  # wait time

    def update_rgbd(self):
        # Read next stream frame in a daemon thread
        while True:
            # Get color_frame and depth frame
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert image to numpy array
            self.imgs[0] = np.asanyarray(color_frame.get_data())
            self.depths[0] = np.asanyarray(depth_frame.get_data())
            time.sleep(1 / self.fps)  # wait time

    def __iter__(self):
        return self

    def __next__(self):
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord("q"):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        if self.depth:
            return img, self.depths[0], img0[0]

        return img, None, img0[0]

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadImages:
    """
    This class load images from the specified folders
    """

    def __init__(
        self, path="output/hello", img_size=640, stride=64, offset=0, color_folder="color", depth_folder="depth"
    ):
        self.img_size = img_size
        self.stride = stride
        self.cont = offset
        self.path = path
        self.color_folder = color_folder
        self.depth_folder = depth_folder
        assert os.path.exists(
            os.path.join(self.path, self.color_folder)
        ), "Not color images folder exist, check dataset structure"
        assert os.path.exists(
            os.path.join(self.path, self.depth_folder)
        ), "Not depth images folder exist, check dataset structure"

        self.color_files = os.listdir(os.path.join(self.path, self.color_folder))
        self.depth_files = os.listdir(os.path.join(self.path, self.depth_folder))
        self.color_files.sort()
        self.depth_files.sort()

        # check for common shapes
        img0 = cv2.imread(os.path.join(self.path, self.color_folder, self.color_files[self.cont]))
        s = np.stack([letterbox(img0, self.img_size, stride=self.stride)[0].shape], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print(
                "WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams."
            )

    def __iter__(self):
        return self

    def __next__(self):
        try:
            img0 = cv2.imread(os.path.join(self.path, self.color_folder, self.color_files[self.cont]))
            depth_image = cv2.imread(
                os.path.join(self.path, self.depth_folder, self.depth_files[self.cont]),
                cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH,
            )
            self.cont += 1
        except IndexError:
            return None, None, None

        # Letterbox
        img = [letterbox(img0, self.img_size, auto=self.rect, stride=self.stride)[0]]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)
        return img, depth_image, img0

    def __len__(self):
        return len(self.color_files)


class CreateRsDataset:
    """
    This class is used to save rgb and depth .png images from the realsense camera
    at the desired fps. It saves the rgb-depth images in the corresponding
    dataset_path/gesture/example_number/color-depth folder
    """

    def __init__(self, dataset_path, gesture, fps=30, color_folder="color", depth_folder="depth"):
        self.dataset = LoadRs()
        base_path = dataset_path
        self.flag = False
        self.index = 0
        self.time_interval = 1.0 / fps
        self.time_ant = None
        self.color_folder = color_folder
        self.depth_folder = depth_folder

        if ensure_exist(os.path.join(base_path, gesture)):
            examples = [int(i) for i in os.listdir(os.path.join(base_path, gesture))]
            if examples:
                new_example = max(examples) + 1
            else:
                new_example = 1
        else:
            new_example = 1
        self.folder = os.path.join(base_path, gesture, str(new_example).zfill(5))
        ensure_exist(os.path.join(self.folder, self.color_folder))
        ensure_exist(os.path.join(self.folder, self.depth_folder))

        self.time_ant = time.time()

    def __wait_time_interval(self):
        now = time.time()
        if self.time_interval - (now - self.time_ant) > 0:
            time.sleep(self.time_interval - (now - self.time_ant))
        self.time_ant = time.time()

    def __call__(self):
        try:
            self.__wait_time_interval()
            _, depth, img0 = next(self.dataset)
        except StopIteration:
            return False
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("img_color", img0)
        cv2.imshow("img_depth", depth_image)
        cv2.waitKey(1)
        if self.flag:
            cv2.imwrite(os.path.join(self.folder, self.color_folder, str(self.index).zfill(5) + ".png"), img0)
            cv2.imwrite(os.path.join(self.folder, self.depth_folder, str(self.index).zfill(5) + ".png"), depth)
            self.index += 1
        return True
