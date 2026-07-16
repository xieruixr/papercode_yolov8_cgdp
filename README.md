# YOLOv8-CGDP Reproducibility

This repository contains the source code needed to reproduce the YOLOv8-CGDP model used in the paper.

## Model

The YOLOv8-CGDP architecture is defined in:

```text
config_model_yaml/yolov8_CG-C2F__DGAF_PCLA_p234_dyhead-detect.yaml
```

The model depends on these custom modules:

- `C2f_ConvFormerCGLU` in `ultralytics/nn/modules/block.py`
- `DGAF` in `ultralytics/nn/modules/block.py`
- `PCLA` in `ultralytics/nn/modules/PCLA.py`
- `Detect_DyHead` in `ultralytics/nn/modules/head.py`

## Installation

Create a Python environment with a PyTorch build that matches your CUDA version, then install the package:

```bash
git clone https://github.com/xieruixr/papercode_yolov8_cgdp.git
cd papercode_yolov8_cgdp
pip install -e .
pip install -r requirements-cgdp.txt
```

YOLOv8-CGDP uses MMCV deformable convolution operators. If your `mmcv` wheel does not include compiled ops for your PyTorch/CUDA version, install MMCV with OpenMIM:

```bash
pip install -U openmim
mim install mmcv
```

## Dataset

Datasets are not included. Prepare the dataset locally and update the `path` field in the dataset YAML.

For VisDrone-style experiments, start from:

```text
dataset_yaml/VisDrone.yaml
```

## Training

```bash
python train_cgdp.py --data dataset_yaml/VisDrone.yaml --epochs 200 --imgsz 640 --batch 8
```

The equivalent Ultralytics CLI command is:

```bash
yolo detect train model=config_model_yaml/yolov8_CG-C2F__DGAF_PCLA_p234_dyhead-detect.yaml data=dataset_yaml/VisDrone.yaml epochs=200 imgsz=640 batch=8 optimizer=SGD
```

## Validation

```bash
python val_cgdp.py --weights runs/detect/train/weights/best.pt --data dataset_yaml/VisDrone.yaml --imgsz 640 --batch 8
```
