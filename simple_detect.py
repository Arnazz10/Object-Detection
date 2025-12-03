import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple object detection using a pretrained YOLO model (no training needed)."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Source for detection: '0' for webcam, or path to image/video/folder.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained YOLO weights to use (downloaded automatically if missing).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load pretrained model (e.g. on COCO dataset)
    model = YOLO(args.model)

    # If source is a digit string like "0", treat it as webcam index
    source: str | int
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    # Run prediction with built‑in Ultralytics UI
    results = model.predict(source=source, conf=args.conf, show=True, save=True)

    # Print a short summary
    for r in results:
        print(f"\nDetections for {Path(getattr(r, 'path', 'source'))}:")
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"  class_id={cls_id} conf={conf:.2f}")


if __name__ == "__main__":
    main()


