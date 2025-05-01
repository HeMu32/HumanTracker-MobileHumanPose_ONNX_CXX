<!-- Language Switch -->
[English](#english-version) | [中文](#中文说明)

---

# 中文说明

基于C++的, 使用了 Yolo-Fastest-v2 以及 Mobile Human Pose 的人物跟踪程序.

使用Yolo检测人. 跟踪主要基于检测框.  
使用稀疏光流和动量辅助跟踪.  
使用Mobile Human Pose进行人体姿态估计. 

# 尝试使用C++推理Mobile Human Pose

原方案请参考: https://github.com/HeMu32/ONNX-Mobile-Human-Pose-3D

# 框架来自: yolo-fastestv2-opencv  
请参看: https://github.com/hpc203/yolo-fastestv2-opencv

使用OpenCV部署Yolo-FastestV2，包含C++和Python两种版本的程序

根据运行体验，这套程序的运行速度真的很快，而且模型文件也很小，可以直接上传到仓库里，  
不用再从百度云盘下载的。

---

# <a name="english-version"></a>English Version

[中文](#中文说明) | [English](#english-version)

---

## C++-based Human Tracking using Yolo-Fastest-v2 and Mobile Human Pose

- Uses Yolo for human detection. Tracking is mainly based on detection bounding boxes.
- Sparse optical flow and momentum are used to assist tracking.
- Mobile Human Pose is used for human pose estimation.

## Try C++ Inference for Mobile Human Pose

Original solution reference: https://github.com/HeMu32/ONNX-Mobile-Human-Pose-3D

## Framework Reference: yolo-fastestv2-opencv  
See: https://github.com/hpc203/yolo-fastestv2-opencv

- Deploys Yolo-FastestV2 with OpenCV, supporting both C++ and Python versions.
- According to actual experience, this program runs very fast and the model files are small enough to be uploaded directly to the repository,  
  so you don't need to download them from Baidu Netdisk.

---