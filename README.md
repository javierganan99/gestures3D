# Gesture Recognition using RGBD Intel Realsense Camera

This repository contains code for 3D gesture recognition using a RGBD intel realsense camera, the YOLOv7 model for skeleton detection, and st-gcn for action recognition.

## Introduction

This project aims to recognize human gestures in a 3D space using an RGBD intel realsense camera. We use the YOLOv7 pose model for skeleton detection and st-gcn for action recognition.

- Skeleton detection with YOLOv7 is performed with the model presented in the [YOLOv7 repository](https://github.com/WongKinYiu/yolov7), and the weights can be downloaded [here](https://drive.google.com/file/d/1KPu864GqracT9QjiWED-X85kte5T8-1x/view?usp=share_link)

- The model for skeleton-based action recognition is an Spatial Temporal Graph Convolutional Network (ST-GCN) based on the one presented at the paper [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455). The model implementation is based on the official implementation of the previous paper in [mmskeleton](https://github.com/open-mmlab/mmskeleton) under the [Apache License](https://github.com/open-mmlab/mmskeleton/blob/master/LICENSE).

The pipeline for the gesture recognition is as follows:

1. Use the RGBD camera to capture a sequence of rgb and depth frames.
2. Extract the skeleton information from each rgb frame using the YOLOv7 model.
3. Combine the 2D skeleton data and the depth to get the 3D skeleton. 
4. Convert the 3D skeleton information into a graph representation.
5. Feed the graph representation into the ST-GCN model to recognize the gesture.

## Aircraft Marshalling Signals Example
<img src="assets/gestures_2.gif" alt="Aircraft Marshalling Signals" style="width:700px">
<img src="assets/gestures_1.gif" alt="Aircraft Marshalling Signals" style="width:700px">



## Installation

1. Clone the repository:

```
git clone https://github.com/javierganan99/gestures3D.git
```

2. Download the docker container image (recommended):

```
sudo docker pull fjganan/gestures3d:latest
```

3. Navigate to the souce folder of the project:

```
cd your_path_to_repo_parent_folder/gestures3D
```

4. Download the pre-trained models:

```
mkdir weights && cd weights
```

[YOLOv7 pose model](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) for skeleton detection
```
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KPu864GqracT9QjiWED-X85kte5T8-1x' -O yolov7-w6-pose.pt
```

[ST-GCN trained](https://github.com/javierganan99/gestures3D/files/11475314/model_gestures.zip) for 6 aircraft marshalling signals
```
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ivZv5z8ZgFCgD17OSHm8Ezms9ACWAnMu' -O model_gestures
```

## Usage

### Docker activation

1. Navigate to the souce folder of the project:

```
cd your_path_to_repo_parent_folder/gestures3D
```


2. Run de docker container corresponding to the image:

```
xhost + # Grant permission to any client to connect to the X server
```

```
sudo docker run -it --rm --privileged -v /dev:/dev -v .:/action3d -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix fjganan/gestures3d
```

**Note**: To use your gpu you could add the option "--gpus all" to the previous command. To connect it with docker, you could install the **NVIDIA Container Toolkit**. Check the following link for instructions: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide.
The docker container includes the driver for the intel realsense camera. If you don´t use the container, check the following link for the driver installation: https://github.com/IntelRealSense/librealsense.git 


### Gesture detection

To predict from RGB and Depth images to gestures:

```
python3 gestures_pedict.py --source rs
```

**--source** argument could be a **folder** containing both "color" and "depth" folders (containing rgb and depth images, respectively), or **rs** to perform inference directly using an intel realsense camera. 

Notice that, when using the realsense on real time, the output rate of the **YOLOv7 pose model** should be similar to the frame rate at which the **ST-GCN** model was trained for accurate gesture prediction. The **ST-GCN** model presented in the **Instalation** section was trained with frames at 30 FPS.

### Demo

If you want to try the aircraft-marshaling-6-signals pretrained model with color and depth images from the realsense, download this sample_images folder:

1. Download the sample images in the following link:

https://drive.google.com/file/d/1QQ2xgk1E99jJ1i2c0OtdvBKQkYMsd2Vr/view?usp=share_link

2. Go to the folder where the zip file is downloaded and unzip it to the repository folder:

```
cd your_path_to/Downloads && unzip sample_images.zip -d your_path_to_repo_parent_folder/gestures3D
```

3. Try the gesture prediction on any downloaded gesture folder:

```
python3 gestures_predict.py --source sample_images/next_left/00001
```


### Custom dataset creation

1. Plug in your realsense rgbd camera.

2. Run save_dataset.py to load the color and depth images from the realsense and store them.
 
    ```
    python3 save_dataset.py --dataset-path dataset_images --gesture your_gesture
    ```

    Place yourself in the camera FOV
    
    Press "S" key to start the recording and "Q" key to stop it. Press "ctrl + C" to exit.
    
    Repeat this step for all the gestures you want to record, considering that **dataset_images** is the output path for your dataset and **your_gesture** is the gesture you are saving

3. Annotate each of the gestures previously recorded:

```
python3 detect_skeleton.py --source dataset_images --view-img --annot
```
This command annotates all the previously recorded gestures. **--view-img** shows the images while detecting the skeleton. **--annot** annotates the gestures inside the folder in which the gesture is stored.

4. Split annotated gestures with desired maximun duration

```
python3 split_gestures_into_chunks.py --dataset-path your_path --output-folder your_output_path --max-frames 120 --stride 3
```

From a dataset **--dataset-path** with gestures annotated, it samples the gestures to create a new dataset at **--output-folder** with the same gestures but with a maximun duration of **--max-frames** frames and sampled with **--stride** 3.

5. From the previous dataset split into class-named folders, create a new dataset divided into train and val folders.

```
python3 train_val_split.py --dataset-path your_path --output-folder your_output_path
```

**--dataset-path** is the input class-split dataset path, **--output-folder** is the path of the output dataset containing the same data but splitted into train and val folders. **--val-size** is the p.u. of the total examples considered for validation.

6. Finally convert the .json skeleton dataset to the proper numpy format of the input of the model.

```
python3 dataset_to_numpy.py --dataset-path your_path --output-folder your_output_path
```

**--dataset-path** is the .json skeleton dataset path, **--output-folder** is the folder in which to save the .npy dataset. **--show** could also be used to show the data during conversion.

### ST-GCN model training

In order to train yout custom ST-GCN model for gesture prediction:

1. Configure the following configuration files inside the *cfg* folder:
    
    - *classes.yaml*: Indicate the classes of your dataset. 
    - *train.yaml*: Indicate the path of your dataset in *.npy* and *.pkl* format.

2. Start the training:

    ```
    python3 gestures_train.py
    ```

3. You can also monitor your training with tensorboard by running:

    ```
    tesorboard --logdir runs/
    ```
4. When the training finished, some models will be output to the folder in which you executed the training. 

