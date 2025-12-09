## Simple Object Detection: Person / Mouse / pen :

This project is a **minimal YOLO-based object detector** template that can detect and classify:

- **person**
- **mouse**
- **pen**

You provide a small custom dataset, and this repo gives you ready-to-run **training** and **inference** scripts.

---

### 1. Install dependencies :

From the project root:

```bash
cd /home/arnab/Documents/CODE/ML/Objdetection-simple

# 1) Install PyTorch first (follow official instructions for your system / GPU)
#    See: https://pytorch.org/get-started/locally/

# 2) Then install the remaining packages
pip install -r requirements.txt
```

---

### 2. Prepare your dataset (YOLO format)

Expected structure:

```text
datasets/
  people-mouse-pen/
    images/
      train/
        img_001.jpg
        img_002.jpg
        ...
      val/
        img_101.jpg
        img_102.jpg
        ...
    labels/
      train/
        img_001.txt
        img_002.txt
        ...
      val/
        img_101.txt
        img_102.txt
        ...
```

- **Each `.txt` file** next to an image has one line per object:

```text
<class_id> <x_center> <y_center> <width> <height>
```

where all coordinates are **normalized** to \([0, 1]\).

- Class IDs for this project:
  - `0` → **person**
  - `1` → **mouse**
  - `2` → **pen**

The file `data-people-mouse-pen.yaml` already points to this dataset layout.

---

### 3. Train the model

Once your dataset is in place:

```bash
cd /home/arnab/Documents/CODE/ML/Objdetection-simple

python train_yolo.py \
  --data data-people-mouse-pen.yaml \
  --model yolo11n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

- This will create a run under something like:
  - `runs/train/people-mouse-pen/`
- The best weights will be in:
  - `runs/train/people-mouse-pen/weights/best.pt`

You can reduce `--epochs` or `--imgsz` if you just want a quick test.

---

### 4. Run detection (images / video)

After training, run inference:

```bash
python detect_yolo.py \
  --weights runs/train/people-mouse-pen/weights/best.pt \
  --source path/to/your/image_or_video \
  --conf 0.25 \
  --show \
  --save
```

- **`--source`** can be:
  - a single image file
  - a directory of images
  - a video file
- **`--show`**: open a window with detections (press any key to close).
- **`--save`**: save annotated images with `_det` appended to filename.

---

### 5. Adapting classes (optional)

If you want different classes:

- Change the `names:` section in `data-people-mouse-pen.yaml`.
- Update `CLASS_NAMES` in `detect_yolo.py` to match your new classes.
- Re‑train the model.

---

### 6. Quick recap

- **You**: collect and label images of people, mice, and pens in YOLO format.
- **This repo**: gives you
  - `train_yolo.py` → train a small YOLO model.
  - `detect_yolo.py` → run detection and visualize/save results.

Once your dataset is ready, you can have a working detector with just **two commands** (train, then detect).


