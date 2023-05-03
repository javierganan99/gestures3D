FROM pytorch/pytorch:latest

LABEL maintainer="fjganan fjganan14@gmail.com"

WORKDIR /action3d

RUN apt-get update && apt-get install -y git cmake libglib2.0-0 libgl1-mesa-glx libusb-1.0-0 libsm6 udev

# Librealsense instalation

RUN apt-get install -y libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev

RUN apt-get install -y libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at

COPY . .

# Udev rules

RUN cp librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/

RUN cp librealsense/config/99-realsense-d4xx-mipi-dfu.rules /etc/udev/rules.d/

# RUN udevadm control --reload-rules 

# RUN udevadm trigger

RUN cd librealsense && mkdir build && cd build && cmake .. && make uninstall && make clean && make && make install

RUN pip install --no-cache-dir -r requirements.txt

# To launch on linux
# xhost +
# sudo docker run -it --rm --gpus all --privileged -v /dev:/dev -v .:/action3d -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix action3d