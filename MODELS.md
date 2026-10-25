# AI models (not in this repository)

The pre-trained model files are third-party, large (about 130 MB) and under their own licences, so they aren't committed.
The app and server run without them: depth scanning, door/furniture outlines and the scene checks switch off, and everything else works.
To get the full system, put each file at the path shown.

| Model | Path | Where to get it | Licence |
|---|---|---|---|
| Depth Anything V2 Small (Core ML, F16) | `ios/Sightline/Models/DepthAnythingV2SmallF16.mlpackage` | Hugging Face: `apple/coreml-depth-anything-v2-small` | Apache-2.0 |
| YOLOv8n, Open Images V7 | `ios/Sightline/Models/YOLOv8nOIV7.mlpackage` | Export from Ultralytics (below) | AGPL-3.0 |
| SegFormer-B2, ADE20K | `server/models/SegFormerADE.mlpackage` | Convert with `tools/convert_segformer.py` (below) | NVIDIA SegFormer licence (non-commercial) |
| MobileNetV2 (backup furniture check) | `server/models/MobileNetV2.mlmodel` | Apple Core ML models page: developer.apple.com/machine-learning/models | Apple |

## Depth Anything V2 Small

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download apple/coreml-depth-anything-v2-small --include "DepthAnythingV2SmallF16.mlpackage/*" --local-dir ios/Sightline/Models
```

## YOLOv8n (Open Images V7)

```bash
pip install ultralytics
yolo export model=yolov8n-oiv7.pt format=coreml nms=True imgsz=640
mv yolov8n-oiv7.mlpackage ios/Sightline/Models/YOLOv8nOIV7.mlpackage
```

## SegFormer-B2

```bash
python3 -m venv tools/convert-venv
tools/convert-venv/bin/pip install torch transformers coremltools numpy
tools/convert-venv/bin/python tools/convert_segformer.py   # writes server/models/SegFormerADE.mlpackage
```

## MobileNetV2

Download `MobileNetV2.mlmodel` from Apple's Core ML models page and save it as `server/models/MobileNetV2.mlmodel`.
