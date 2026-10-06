# Few-shot Labeler

PyQt5 app for labeling bounding boxes from a few examples. Draw a box or two, and it suggests the rest. No training. Exports YOLO labels.

**How it works:** DINOv2 patch features find similar objects (rotation-augmented matching), then SAM box prompts tighten each box.

## Install

```
uv add PyQt5 opencv-python-headless numpy transformers torch
```

Use `opencv-python-headless`, not `opencv-python`, to avoid Qt plugin conflicts.

## Run

```
uv run labeler.py
```

Models download on first use to the Hugging Face cache (`~/.cache/huggingface`, about 465 MB total).

## Usage

1. Open a folder of images and add a class.
2. Drag tight boxes around a few examples.
3. Go to the next image. Dashed boxes are suggestions. Click one to accept it, right-click to delete, `Enter` to accept all.
4. `Ctrl+S` saves (also on close) to `<folder>/labels/` as YOLO `.txt` files plus `classes.txt`.

Keys: `1-9` class, `←/→` image, `S` suggest, `Enter` accept all.
Raise the threshold slider for fewer false boxes, lower it for more.

## Troubleshooting

- `WinError 1114` on Windows: keep `import torch` before the PyQt5 imports.
- `sam_unavailable` in the log: SAM failed to load, and the app falls back to plain window boxes.

## Limits

Boxes use the median example size, so draw tight examples. Only boxes are exported. Runs on the UI thread, so it may freeze briefly on CPU.