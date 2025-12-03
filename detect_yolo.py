import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


CLASS_NAMES = {0: "person", 1: "mouse", 2: "pen"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a YOLO model for person/mouse/pen."
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained weights (e.g. runs/train/people-mouse-pen/weights/best.pt).",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image, directory of images, or video file.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If set, show image/video window with detections.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="If set, save annotated outputs next to the inputs.",
    )
    return parser.parse_args()


def draw_and_save(result, source_path: Path, save: bool, show: bool) -> None:
    img = result.plot()  # BGR numpy array with boxes & labels drawn

    if save:
        if source_path.is_file():
            out_path = source_path.with_name(f"{source_path.stem}_det{source_path.suffix}")
        else:
            out_path = Path("detections.jpg")
        cv2.imwrite(str(out_path), img)
        print(f"Saved: {out_path}")

    if show:
        cv2.imshow("detections", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path.resolve()}")

    model = YOLO(str(weights_path))

    results = model(args.source, conf=args.conf)

    for result in results:
        source_path = Path(result.path)
        print(f"\nDetections for {source_path}:")
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = CLASS_NAMES.get(cls_id, str(cls_id))
            print(f"  - {name} (class {cls_id}) | conf={conf:.2f}")

        draw_and_save(result, source_path, save=args.save, show=args.show)


if __name__ == "__main__":
    main()


