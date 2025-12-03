import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple YOLO object detector for person/mouse/pen."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data-people-mouse-pen.yaml",
        help="Path to the dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Base YOLO model to fine‑tune (e.g. yolo11n.pt, yolov8n.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (pixels).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Directory to save training runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="people-mouse-pen",
        help="Run name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path.resolve()}\n"
            "Make sure you've created your dataset and the YAML file."
        )

    # Load a small YOLO model (change to a different checkpoint if you prefer)
    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()


