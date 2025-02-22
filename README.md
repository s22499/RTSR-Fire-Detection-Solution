# RTSR-Fire-Detection-Solution
Computer Vision dataset and scripts to detect fire and smoke in server room enviroment, using YOLOv8. 

# Abstract
The aim of the study is to develop a computer vision (CV) model for detecting fire
and smoke in server rooms, considering specific industrial conditions. The implementation is based on the YOLOv8 model, the PyTorch library, and the Roboflow platform for
manual data annotation. The model has been adapted to variable lighting conditions,
different camera resolutions, and real-time detection requirements.

A key step was the creation of an appropriate dataset. Data augmentation techniques were also tested, including brightness adjustments (±25%), rotation (up to 15°),
and mirroring. To achieve a balance between speed and accuracy, the YOLOv8 Medium model was selected. Fire detection achieved a precision of 89.2% and an mAP50
score of 88.7%. The results for the "smoke" class were weaker, attributed to the model’s
sensitivity to variable camera parameters such as lighting and contrast.

The study also includes three scripts enabling detection in various scenarios: video
files, image sequences, and real-time detection using USB cameras. These scripts allow
for further research and the implementation of the solution in a production environment.

### keywords: computer vision, YOLOv8, PyTorch, Roboflow, data augmentation, fire detection, smoke detection, real-time inference, server room security

# Roboflow dataset
#### https://app.roboflow.com/server-room-fire-and-smoke-detection/serveroom-fire-and-smoke-dtc./browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
# Prerequisites to use scripts

* Ultralytics: https://docs.ultralytics.com/quickstart/
* PyTorch: https://pytorch.org
* GPU with NVIDIA-CUDA
* Python
* Open-CV: https://opencv.org

Versions of libraries used during project making:
* Ultralytics: 8.2.19
* PyTorch: 2.3.0
* CUDA-API: 12.1.0

## Quickinstall

### Check CUDA Version
Use: nvidia-smi in PowerShell

![image](https://github.com/user-attachments/assets/f2754d73-debf-476d-9171-d0dbad08e537)


### Install all packages together using conda
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=your_cuda_version ultralytics

### Install the ultralytics package from PyPI
pip install ultralytics
