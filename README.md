# Crowd Clustering Detection Dashboard

This repository now contains a production-oriented web dashboard that turns the original notebook-based crowd clustering workflow into a reusable application for uploaded video analysis.

## What It Does

- Uploads local `MP4`, `AVI`, or `MOV` files
- Runs person detection frame-by-frame with Ultralytics YOLO
- Tracks people over time with ByteTrack
- Normalizes motion with perspective-aware spatial scaling
- Marks only stationary groups as clusters
- Ignores moving pedestrian flows
- Streams annotated frames live in the dashboard
- Exposes real-time analytics for people, clusters, stationary individuals, and processing FPS
- Exports a fully annotated processed `MP4`

## Clustering Rules

A group is counted as a cluster only when:

- At least `3` people are present
- Each person remains within a configurable spatial radius for a configurable duration
- The group is stationary rather than walking together
- Perspective normalization is applied so far-away motion is not underestimated


```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Notes

- The default detector is `yolov8n.pt`. Change it in the dashboard form if you want a larger model such as `yolov8m.pt` or `yolov8l.pt`.
- If the selected model weights are not already present locally, Ultralytics may download them the first time the app runs.
- Processed videos and temporary runtime artifacts are written under `runtime/`.

## Validation

The core clustering heuristics are covered by tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_cluster_logic.py
```


