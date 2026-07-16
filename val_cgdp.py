"""Minimal validation entry point for YOLOv8-CGDP weights."""

import argparse
import warnings

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLOv8-CGDP weights.")
    parser.add_argument("--weights", required=True, help="Path to trained YOLOv8-CGDP weights.")
    parser.add_argument("--data", default="dataset_yaml/VisDrone.yaml", help="Path to the dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size.")
    parser.add_argument("--split", default="val", help="Dataset split to validate.")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0 or 0,1. Uses Ultralytics default if omitted.")
    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    model = YOLO(args.weights)
    val_kwargs = vars(args).copy()
    val_kwargs.pop("weights")
    if val_kwargs["device"] is None:
        val_kwargs.pop("device")
    model.val(**val_kwargs)


if __name__ == "__main__":
    main()
