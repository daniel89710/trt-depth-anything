# TensorRT Depth Anything for Effcient Inference

## Purpose

This package estimates depth for arbitary images using TensorRT Depth Anything for efficient and faster inference.
Specifically, this supports multi-precision and multi-devices inference for efficient inference on embedded platform.

## Setup

Install basic libraries for inference on GPUs including CUDA, cuDNN, TensorRT (>=8.6) and OpenCV.

Moreover, you need to install as bellow.
```bash
sudo apt-get install libgflags-dev
sudo apt-get install libboost-all-dev
```

## ONNX Conversion

Please use the export_to_onnx.py from the following repository for the converter.

https://github.com/spacewalk01/depth-anything-tensorrt

## Build sources

```bash
git clone git@github.com:tier4/trt-depth-anything.git
cd trt-depth-anything.git
cd build/
cmake ..
make -j
```

## Start inference

-Build TensoRT engine

```bash
./trt-depth-anything --onnx depth_anything_vitb14.onnx --precision fp32
```

-Infer from a Video

```bash
./trt-depth-anything --onnx depth_anything_vitb14.onnx --precision fp32 --v {VIDEO PATH} --depth {gray/magma/jet} (--max_depth VALUE)
```

-Infer from images in a directory
```bash
./trt-depth-anything --onnx depth_anything_vitb14.onnx --precision fp32 --d {Directory PATH} --depth {gray/magma/jet} (--max_depth VALUE) (--save_detections --save_detections_path {SAVE_PATH}) (--dont_show)
```


### Cite

Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao, "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data", arXiv:2401.10891, 2024 [[ref](https://arxiv.org/abs/2401.10891)]


## Parameters

--precision={fp32,fp16}

--v={VIDEO PATH}

--d={Directroy PATH}

--depth={gray/magma/jet} : default is 'gray'

--max_depth={VALUE} : option

## Assumptions / Known limits

### Todo

- [ ] Confirm accuracy using INT8
- [ ] Support Multi-batch execution

## Reference repositories

- <https://github.com/LiheYoung/Depth-Anything>
- <https://github.com/spacewalk01/depth-anything-tensorrt>
- <https://github.com/autowarefoundation/autoware.universe/tree/main/common/tensorrt_common>
- <https://github.com/autowarefoundation/autoware.universe/tree/main/common/cuda_utils>
