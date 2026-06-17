# YOLOv8-CGDP Paper Code

This repository contains the code release for YOLOv8-CGDP, a YOLOv8-based object detection model.

The target architecture is:

```text
config_model_yaml/yolov8_DCF-C2F_RGFM_SPECA_p234_dyhead.yaml
```

For complete installation and reproduction instructions, see [README_CGDP.md](README_CGDP.md).

## What Is Included

- YOLOv8-CGDP model YAML.
- Modified Ultralytics source code needed by the model.
- Dataset YAML examples for local dataset preparation.
- Minimal training and validation entry points.

## What Is Not Included

Datasets, trained weights, local training runs, result folders, videos, and cache files are intentionally excluded.

## Quick Start

```bash
pip install -e .
pip install -r requirements-cgdp.txt
python train_cgdp.py --data dataset_yaml/VisDrone.yaml --epochs 200 --imgsz 640 --batch 16
```

## License

This codebase is derived from Ultralytics YOLOv8 and follows the AGPL-3.0 license included in this repository.
