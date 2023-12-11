import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import torch
from utils.auxiliary import VideoSaver, load_yaml, ensure_exist, images_to_video
from utils.transformations import translate_to_center_batch


def split_skeleton_annotation(file, dataset_path="dataset", max_T=120, stride=3):
    """
    This function reads a Skeleton annotation file and split it into chuncks of max_T
    frames,saving it into dataset_path folder inside its corresponding label folder
    """
    with open(file, "r") as f:
        video_info = json.load(f)

    path = os.path.join(dataset_path, video_info["label"])

    if ensure_exist(path):
        examples = [int(i[:-5]) for i in os.listdir(path)]
        if examples:
            new_example = max(examples) + 1
        else:
            new_example = 1
    else:
        new_example = 1

    n_files = (len(video_info["data"]) - max_T) // stride

    for f in range(n_files):
        try:
            data = video_info["data"][f * stride : (f * stride) + max_T]
        except IndexError:
            data = video_info["data"][f * stride :]
        for i, d in enumerate(data):
            d["frame_index"] = i
        new_file = {
            "data": data,
            "label": video_info["label"],
            "label_index": video_info["label_index"],
        }
        with open(
            os.path.join(path, str(new_example).zfill(7) + ".json"), "w"
        ) as outfile:
            outfile.write(json.dumps(new_file, indent=4))
            new_example += 1


class SkeletonBuffer:
    """
    It appends skeleton data to a buffer in numpy format with shape (1,C,T,V,num_person).
        C: Number of channels
        T: Number of consecutive skeletons
        V: Number of keypoints
        num_person: number of persons
    """

    def __init__(self, C=4, T=120, V=17, num_person=1):
        self.C, self.T, self.V, self.num_person = C, T, V, num_person
        self.buffer = np.zeros((1, C, T, V, self.num_person), dtype="float32")

    def append(self, pose, score):
        self.buffer = np.roll(self.buffer, -1, axis=2)
        for i in range(self.C - 1):
            self.buffer[:, i, -1, :, self.num_person - 1] = pose[i::3]
        self.buffer[:, self.C - 1, -1, :, self.num_person - 1] = score
        # Custom Transformations
        self.buffer[0, ...] = translate_to_center_batch(self.buffer[0, ...])


class Skeleton3D:
    """
    It calculates the 3D skeleton from RGB and Depth images. It contains methods
    to plot and annotate the skeleton data. It can also save the predicted skeletons.
    """

    def __init__(
        self,
        annot=False,
        show_images=False,
        annot_name="action",
        cfg_rs="cfg/realsense.yaml",
        cfg_classes="cfg/classes.yaml",
        drawing_file="cfg/drawing.yaml",
        output_path=None,
        color_shape=(640, 512),
        depth_shape=(640, 480),
    ):
        self.show = show_images
        params_rs = load_yaml(cfg_rs)
        self.depth_scale = params_rs["depth_scale"]
        # 3D Pose estimation
        self.A = np.array(params_rs["camera_matrix"]["data"]).reshape(
            params_rs["camera_matrix"]["rows"], params_rs["camera_matrix"]["cols"]
        )
        self.window_size = params_rs["window_size"]
        # The figure to show the 3D skeleton plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="3d")
        self.ax.view_init(elev=90.0, azim=90)
        self.annot = annot
        if self.annot:
            annot_dict = load_yaml(cfg_classes)
            pair = [(k, v) for k, v in annot_dict.items() if v in annot_name][0]
            self.annot_name = annot_name
            self.annotations = {"data": [], "label": pair[0], "label_index": pair[1]}
            self.frame_cont = 0
        # To plot the current gesture
        self.gesture = " "
        # Color and style of the skeleton
        drawing = load_yaml(drawing_file)
        self.palette = np.array(drawing["palette"])
        self.skeleton = drawing["skeleton"]
        self.pose_limb_color = self.palette[drawing["pose_limb_color"]]
        self.pose_kpt_color = self.palette[drawing["pose_kpt_color"]]
        self.gesture_color = tuple(np.array(drawing["gesture_color"]) / 255)
        # To save the skeleton
        if output_path is not None:
            self.output_path = output_path
            self.video_color = VideoSaver(
                path=output_path, size=color_shape, fps=30, color=True, name="color.avi"
            )  # Color video
            self.video_depth = VideoSaver(
                path=output_path, size=depth_shape, fps=30, color=True, name="depth.avi"
            )  # Depth video
            ensure_exist(os.path.join(self.output_path, "plots"))  # Plots folder
        else:
            self.video_color = None
            self.video_depth = None
        self.cont = 0

    def add_frame(self, pts, scores):
        """
        Add skeleton frame to annotations dict
        """
        self.frame_dict = {"frame_index": self.frame_cont, "skeleton": []}
        skeleton = self.add_skeleton(pts, scores)
        self.frame_dict["skeleton"].append(skeleton)
        self.annotations["data"].append(self.frame_dict)
        self.frame_cont += 1

    @staticmethod
    def add_skeleton(pts, scores):
        skeleton = {}
        skeleton["pose"] = pts
        skeleton["score"] = scores
        return skeleton

    def write_annotations(self, filename="annotation.json"):
        json_obj = json.dumps(self.annotations, indent=4)
        with open(filename, "w") as outfile:
            outfile.write(json_obj)

    def calculate_3D_points(self, u, v, Z):
        """
        Calculate 3D points from (u,v) pixels and depth
        """
        X = (u - self.A[0, 2]) / self.A[0, 0] * Z
        Y = (v - self.A[1, 2]) / self.A[1, 1] * Z
        return X, Y, Z

    def saturate(self, x, y, w, h):
        """
        Restrict the points to the image size
        """
        if y > h:
            y = h
        if y < 0:
            y = 0
        if x > w:
            x = w
        if x < 0:
            x = 0
        return x, y

    def depth_windowing(self, depth, x, y):
        """
        Filter the depth to prevent from noise
        """
        mini = 100
        depth = depth * self.depth_scale
        for i in range(-self.window_size, self.window_size):
            for j in range(-self.window_size, self.window_size):
                try:
                    d = depth[y + j, x + i]
                    if d < mini and d != 0:
                        mini = d
                except:
                    pass
        return mini

    def plot_3D_skeleton(self, kpts, confidences, steps=3):
        """
        Plot Skeleton from keypoints
        """
        num_kpts = len(confidences)
        self.ax.cla()
        # To fix the axes
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        self.ax.set_xticks(np.arange(-1, 1, 0.5))
        self.ax.set_yticks(np.arange(-1, 1, 0.5))
        self.ax.set_zticks(np.arange(-1, 1, 0.5))
        # To draw the gesture
        self.ax.text(
            0.75, -0.75, 0.25, self.gesture, color=self.gesture_color, fontsize=20
        )
        for kid in range(num_kpts):
            r, g, b = self.pose_kpt_color[kid]
            if confidences[kid] < 0.5:
                continue
            self.ax.scatter(
                kpts[steps * kid],
                kpts[steps * kid + 1],
                kpts[steps * kid + 2],
                marker="o",
                c=np.array([[int(b / 255), int(g / 255), int(r / 255)]]),
            )

        for sk_id, sk in enumerate(self.skeleton):
            conf1 = confidences[sk[0] - 1]
            conf2 = confidences[sk[1] - 1]
            if conf1 < 0.5 or conf2 < 0.5:
                continue
            r, g, b = self.pose_limb_color[sk_id]
            x = [kpts[(sk[0] - 1) * steps], kpts[(sk[1] - 1) * steps]]
            y = [kpts[(sk[0] - 1) * steps + 1], kpts[(sk[1] - 1) * steps + 1]]
            z = [kpts[(sk[0] - 1) * steps + 2], kpts[(sk[1] - 1) * steps + 2]]
            self.ax.plot(x, y, z, color=(int(b / 255), int(g / 255), int(r / 255)))
        plt.draw()
        plt.pause(0.00000001)

    def plot_skeleton(self, im, depth, kpts, steps):
        """
        Plot Skeleton from keypoints and images
        """
        radius = 5
        num_kpts = len(kpts) // steps

        self.ax.cla()
        # To fix the axes
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        self.ax.set_xticks(np.arange(-1, 1, 0.5))
        self.ax.set_yticks(np.arange(-1, 1, 0.5))
        self.ax.set_zticks(np.arange(-1, 1, 0.5))
        # To draw the gesture
        self.ax.text(
            0.75, -0.75, 0.25, self.gesture, color=self.gesture_color, fontsize=20
        )
        points_3d = []
        confidences = []
        depth_image = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET
        )
        for kid in range(num_kpts):
            r, g, b = self.pose_kpt_color[kid]
            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]

            for i in range(3):
                points_3d.append(0)
            confidences.append(0)

            if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                if steps == 3:
                    conf = kpts[steps * kid + 2]
                    if conf < 0.5:
                        continue
                    confidences[-1] = conf
                x_coord, y_coord = self.saturate(x_coord, y_coord, 639, 479)
                Z = self.depth_windowing(depth, int(x_coord), int(y_coord))
                X, Y, Z = self.calculate_3D_points(x_coord, y_coord, Z)
                points_3d[-3:] = X, Y, Z
                self.ax.scatter(
                    X,
                    Y,
                    Z,
                    marker="o",
                    c=np.array([[int(b / 255), int(g / 255), int(r / 255)]]),
                )
                if self.show:
                    cv2.circle(
                        im,
                        (int(x_coord), int(y_coord)),
                        radius,
                        (int(r), int(g), int(b)),
                        -1,
                    )
                    cv2.rectangle(
                        im,
                        (
                            int(x_coord) - self.window_size,
                            int(y_coord) - self.window_size,
                        ),
                        (
                            int(x_coord) + self.window_size,
                            int(y_coord) + self.window_size,
                        ),
                        (0, 255, 0),
                        2,
                    )
                    cv2.rectangle(
                        depth_image,
                        (
                            int(x_coord) - self.window_size,
                            int(y_coord) - self.window_size,
                        ),
                        (
                            int(x_coord) + self.window_size,
                            int(y_coord) + self.window_size,
                        ),
                        (0, 255, 0),
                        2,
                    )

        if np.all(np.array(confidences) == 0):
            return None

        # Store the annotation
        if self.annot:
            self.add_frame(points_3d, confidences)

        for sk_id, sk in enumerate(self.skeleton):
            r, g, b = self.pose_limb_color[sk_id]
            pos1 = (int(kpts[(sk[0] - 1) * steps]), int(kpts[(sk[0] - 1) * steps + 1]))
            pos2 = (int(kpts[(sk[1] - 1) * steps]), int(kpts[(sk[1] - 1) * steps + 1]))
            x = [points_3d[(sk[0] - 1) * steps], points_3d[(sk[1] - 1) * steps]]
            y = [points_3d[(sk[0] - 1) * steps + 1], points_3d[(sk[1] - 1) * steps + 1]]
            z = [points_3d[(sk[0] - 1) * steps + 2], points_3d[(sk[1] - 1) * steps + 2]]

            if steps == 3:
                conf1 = kpts[(sk[0] - 1) * steps + 2]
                conf2 = kpts[(sk[1] - 1) * steps + 2]
                if conf1 < 0.5 or conf2 < 0.5:
                    continue
            if pos1[0] % 640 == 0 or pos1[1] % 640 == 0 or pos1[0] < 0 or pos1[1] < 0:
                continue
            if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0] < 0 or pos2[1] < 0:
                continue
            self.ax.plot(x, y, z, color=(int(b / 255), int(g / 255), int(r / 255)))
            if self.show:
                cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
        if self.show:
            cv2.putText(
                im,
                self.gesture,
                org=(int(im.shape[1] * 0.5), int(im.shape[0] * 0.15)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=np.array(self.gesture_color) * 255,
                thickness=2,
            )
            cv2.imshow("color", im)
            cv2.imshow("depth", depth_image)
            cv2.waitKey(1)
        if self.video_color is not None:
            self.video_color(im)
            self.video_depth(depth_image)
            plt.savefig(
                os.path.join(
                    self.output_path, "plots", str(self.cont).zfill(5) + ".png"
                )
            )
        plt.draw()
        plt.pause(0.00000001)
        self.cont += 1
        return points_3d, confidences

    def calculate_3D_skeleton(self, depth, kpts, steps=3):
        num_kpts = len(kpts) // steps
        points_3d = []
        confidences = []
        for kid in range(num_kpts):
            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]

            for _ in range(3):
                points_3d.append(0)
            confidences.append(0)

            if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                if steps == 3:
                    conf = kpts[steps * kid + 2]
                    if conf < 0.5:
                        continue
                    confidences[-1] = conf
                x_coord, y_coord = self.saturate(x_coord, y_coord, 639, 479)
                Z = self.depth_windowing(depth, int(x_coord), int(y_coord))
                X, Y, Z = self.calculate_3D_points(x_coord, y_coord, Z)
                points_3d[-3:] = X, Y, Z

        if np.all(np.array(confidences) == 0):
            return None, None

        # Store the annotation
        if self.annot:
            self.add_frame(points_3d, confidences)
        return points_3d, confidences

    def __del__(self):
        if self.video_color is not None:
            images_to_video(os.path.join(self.output_path, "plots"))


class SkeletonFeeder(torch.utils.data.Dataset):
    """
    This class reads the skeleton .json files in data_path folder and returns numpy data.
    """

    def __init__(
        self,
        data_path,
        num_person_in=1,
        num_person_out=1,
    ):
        self.data_path = data_path
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out

        self.load_data()

    def load_data(self):
        # load file list
        self.sample_name = [
            file for file in os.listdir(self.data_path) if file.endswith(".json")
        ]
        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_name)  # samples
        self.C = 4  # channels
        self.T = 120  # frames
        self.V = 17  # joints
        self.M = self.num_person_out  # persons

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # output shape (C, T, V, M)
        # get data
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)
        with open(sample_path, "r") as f:
            video_info = json.load(f)

        # get label index
        label = video_info["label_index"]

        # fill data_numpy
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        for frame_info in video_info["data"]:
            frame_index = frame_info["frame_index"]
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                if m >= self.num_person_in:
                    break
                pose = skeleton_info["pose"]
                score = skeleton_info["score"]
                for i in range(self.C - 1):
                    data_numpy[i, frame_index, :, m] = pose[i::3]
                data_numpy[self.C - 1, frame_index, :, m] = score

        data_numpy = data_numpy[:, :, :, 0 : self.num_person_out]

        return data_numpy, label
