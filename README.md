## Note: Python 2.x support has officially been dropped.

# Berkeley AUTOLAB's GQCNN Package
<p>
   <a href="https://travis-ci.org/BerkeleyAutomation/gqcnn/">
       <img alt="Build Status" src="https://travis-ci.org/BerkeleyAutomation/gqcnn.svg?branch=master">
   </a>
   <a href="https://github.com/BerkeleyAutomation/gqcnn/releases/latest">
       <img alt="Release" src="https://img.shields.io/github/release/BerkeleyAutomation/gqcnn.svg?style=flat">
   </a>
   <a href="https://github.com/BerkeleyAutomation/gqcnn/blob/master/LICENSE">
       <img alt="Software License" src="https://img.shields.io/badge/license-REGENTS-brightgreen.svg">
   </a>
   <a>
       <img alt="Python 3 Versions" src="https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-yellow.svg">
   </a>
</p>

## Package Overview
The gqcnn Python package is for training and analysis of Grasp Quality Convolutional Neural Networks (GQ-CNNs). It is part of the ongoing [Dexterity-Network (Dex-Net)](https://berkeleyautomation.github.io/dex-net/) project created and maintained by the [AUTOLAB](https://autolab.berkeley.edu) at UC Berkeley.

## Installation and Usage
Please see the [docs](https://berkeleyautomation.github.io/gqcnn/) for installation and usage instructions.

## Citation
If you use any part of this code in a publication, please cite [the appropriate Dex-Net publication](https://berkeleyautomation.github.io/gqcnn/index.html#academic-use).

# Guide for RISE members
> (WARNING) Download pre-trained model from our synology. Please see details on bellow information. 
## Installation
Download code on your *catkin_ws*.
```bash
git clone -b melodic-devel --single-branch https://github.com/rise-lab-skku/rise-gqcnn.git
```

Recommanded: Use virtual environment and activate it.
```
cd rise-gqcnn
virtualenv -p python3.6 --system-site-packages venv
source venv/bin/activate
```

Change directories into the *gqcnn* repository and run the pip installation.
```bash
pip install .
```

## Download pre-trained models from our synology
> (WARNING) Official download link is broken. Please follow the bellow intruction.

Create directory in the *gqcnn* repository.
```bash
mkdir -p models
```
Download pre-trained models on *models* directory. The models can be found on our synology `/Research Projects/2020_지능증강/NN_models/official-gqcnn-models`.

Unzip pre-trained models.
```bash
cd models
unzip -a GQCNN-4.0-PJ.zip
unzip -a GQCNN-4.0-SUCTION.zip
unzip -a FC-GQCNN-4.0-PJ.zip
unzip -a FC-GQCNN-4.0-SUCTION.zip
cd ..
```

## Usage
Start the grasp planning service:
```bash
roslaunch gqcnn grasp_planning_service.launch ns:=pj_gqcnn model_name:=FC-GQCNN-4.0-PJ fully_conv:=true
```

The example ROS policy can then be queried on saved images using:
```bash
python examples/policy_ros.py --depth_image data/examples/clutter/phoxi/fcgqcnn/depth_0.npy --segmask data/examples/clutter/phoxi/fcgqcnn/segmask_0.png --camera_intr data/calib/phoxi/phoxi.intr --namespace pj_gqcnn
```

# GQ-CNN with PyTorch
This branch contains a PyTorch implementation of the GQ-CNN, where you can use the original TensorFlow implementation on this branch. The original TensorFlow implementation can be found on the [melodic-devel](https://github.com/rise-lab-skku/rise-gqcnn/tree/melodic-devel).

## Additional Installation
Install PyTorch 1.10.1 with CUDA 11.1. You can install other versions of CUDA, but you need to install the corresponding version of PyTorch. Please see [PyTorch official website](https://pytorch.org/get-started/locally/) for more details.
```bash
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

## Download pre-trained models from our synology
Download pre-trained models on *models* directory. The models can be found on our synology `/Research Projects/2020_지능증강/NN_models/pytorch-gqcnn-models`.

## Usage
Start the grasp planning service:
```bash
source /home/sungwon/ws/ros_ws/bin_picking_ws/src/grasp_estimations/rise-gqcnn/venv/bin/activate
roslaunch gqcnn_ros grasp_planning_service.launch ns:=pj_gqcnn model_name:=PYTORCH-GQCNN-4.0-PJ backend:=pytorch
```

```bash
source /home/sungwon/ws/ros_ws/bin_picking_ws/src/grasp_estimations/rise-gqcnn/venv/bin/activate
roslaunch gqcnn_ros grasp_planning_service.launch ns:=sc_gqcnn model_name:=PYTORCH-GQCNN-4.0-SUCTION backend:=pytorch
```
