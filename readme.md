# rknn object detection demo

## requirements

rknn-toolkit2 virtual environment is required

### Installation

Instructions are taken from the pip instalaltion section from this file
https://github.com/airockchip/rknn-toolkit2/blob/master/doc/02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V2.3.2_EN.pdf

```bash
# Install conda
wget -c https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
chmod 777 Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh
# Activate conda
source ~/miniforge3/bin/activate
# Create venv for RKNN Toolkit2
conda create -n RKNN-Toolkit2 python=3.8
conda activate RKNN-Toolkit2
pip install rknn-toolkit2 -i https://pypi.org/simple
```

## activate conda

Put this snipped on a `.sh` file to activate the environment

```bash
#! /bin/bash
cd
source miniforge3/bin/activate
conda activate RKNN-Toolkit2
```

## run on camera or video

You can use the script with the following commands

```bash
python cam_classes.py --camera_id
python cam_classes.py --camera_id 50 # camera id 50, run v4l2-ctl --device-list
python cam_classes.py --camera_id "ppvideo.mp4" # add the route of a video
```

# intel CPU requirements

```bash
pip cache purge
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy opencv-python onnxruntime ultralytics --no-deps
```
