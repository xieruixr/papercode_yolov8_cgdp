"""Minimal training entry point for YOLOv8-CGDP."""

import argparse
import warnings

from ultralytics import YOLO


DEFAULT_MODEL = "config_model_yaml/yolov8_DCF-C2F_RGFM_SPECA_p234_dyhead.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8-CGDP.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to the YOLOv8-CGDP model YAML.")
    parser.add_argument("--data", default="dataset_yaml/VisDrone.yaml", help="Path to the dataset YAML.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--optimizer", default="SGD", help="Optimizer name passed to Ultralytics.")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0 or 0,1. Uses Ultralytics default if omitted.")
    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    model = YOLO(args.model)
    train_kwargs = vars(args).copy()
    train_kwargs.pop("model")
    if train_kwargs["device"] is None:
        train_kwargs.pop("device")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
