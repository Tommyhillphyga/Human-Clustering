from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np
import torch

from app.config import TMP_DIR
from app.schemas import ProcessingConfig
from app.services.cluster_logic import AnalysisFrame, CrowdClusterAnalyzer

TMP_DIR.mkdir(parents=True, exist_ok=True)
MPL_DIR = TMP_DIR / "matplotlib"
YOLO_DIR = TMP_DIR / "ultralytics"
CACHE_DIR = TMP_DIR / "cache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
YOLO_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


class CrowdVideoProcessor:
    def __init__(self, config: ProcessingConfig) -> None:
        self.config = config.sanitized()
        self.model = YOLO(self.config.model_name)
        if self.config.device:
            self.model.overrides["device"] = self.config.device

    def process(self, input_path: Path, output_path: Path, job) -> None:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Unable to create output video: {output_path}")

        tracker = self._build_tracker(fps)
        analyzer = CrowdClusterAnalyzer(self.config, fps=fps)
        fps_samples: list[float] = []

        job.mark_started(total_frames=total_frames, output_path=str(output_path))

        try:
            frame_index = 0
            while True:
                job.wait_if_paused()
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1

                started = time.perf_counter()
                detections = self._detect_people(frame)
                tracked = tracker.update(detections, img=frame)
                analysis = analyzer.analyze(tracked, frame_index=frame_index, frame_shape=frame.shape)
                processing_fps = self._rolling_fps(started, fps_samples)

                annotated = self._annotate_frame(
                    frame=frame.copy(),
                    analysis=analysis,
                    processing_fps=processing_fps,
                    frame_index=frame_index,
                    total_frames=total_frames,
                )
                writer.write(annotated)

                encoded, frame_buffer = cv2.imencode(
                    ".jpg",
                    annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82],
                )
                frame_bytes = frame_buffer.tobytes() if encoded else None

                job.update_frame(
                    frame_index=frame_index,
                    total_frames=total_frames,
                    processing_fps=processing_fps,
                    analysis=analysis,
                    frame_bytes=frame_bytes,
                )

            job.mark_completed()
        finally:
            cap.release()
            writer.release()

    def _detect_people(self, frame: np.ndarray) -> Boxes:
        result = self.model.predict(
            frame,
            imgsz=self.config.imgsz,
            conf=self.config.conf_thresh,
            iou=self.config.iou_thresh,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None:
            return Boxes(torch.empty((0, 6), dtype=torch.float32), frame.shape[:2]).cpu().numpy()

        if len(boxes) == 0:
            return boxes.cpu().numpy()

        return boxes[boxes.cls == 0].cpu().numpy()

    def _build_tracker(self, fps: float) -> BYTETracker:
        args = SimpleNamespace(
            track_buffer=self.config.track_buffer,
            track_high_thresh=self.config.track_high_thresh,
            track_low_thresh=self.config.track_low_thresh,
            new_track_thresh=self.config.new_track_thresh,
            match_thresh=self.config.match_thresh,
            fuse_score=True,
        )
        return BYTETracker(args=args, frame_rate=int(fps) or 30)

    def _rolling_fps(self, started: float, fps_samples: list[float]) -> float:
        elapsed = max(time.perf_counter() - started, 1e-6)
        fps = 1.0 / elapsed
        fps_samples.append(fps)
        if len(fps_samples) > 30:
            fps_samples.pop(0)
        return float(sum(fps_samples) / len(fps_samples))

    def _annotate_frame(
        self,
        frame: np.ndarray,
        analysis: AnalysisFrame,
        processing_fps: float,
        frame_index: int,
        total_frames: int,
    ) -> np.ndarray:
        cluster_map = {}
        for cluster in analysis.clusters:
            for track_id in cluster.member_track_ids:
                cluster_map[track_id] = cluster.cluster_id

        overlay = frame.copy()
        for cluster in analysis.clusters:
            cv2.circle(overlay, cluster.centroid_pixel, cluster.radius_px, (20, 36, 220), 3)
            label_position = (
                max(cluster.centroid_pixel[0] - cluster.radius_px, 12),
                max(cluster.centroid_pixel[1] - cluster.radius_px - 12, 28),
            )
            cv2.putText(
                overlay,
                f"Cluster {cluster.cluster_id}",
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (20, 36, 220),
                2,
                cv2.LINE_AA,
            )
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        for person in analysis.persons:
            if person.track_id in cluster_map:
                color = (20, 36, 220)
                status_label = f"cluster {cluster_map[person.track_id]}"
            elif person.is_stationary:
                color = (0, 168, 255)
                status_label = "stationary"
            else:
                color = (72, 201, 112)
                status_label = "moving"

            x1, y1, x2, y2 = person.bbox
            for index in range(1, len(person.trail)):
                cv2.line(frame, person.trail[index - 1], person.trail[index], color, 2, cv2.LINE_AA)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {person.track_id} | {status_label} | {person.speed:.1f} u/s"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                cv2.LINE_AA,
            )

        self._draw_metrics_panel(frame, analysis, processing_fps, frame_index, total_frames)
        return frame

    def _draw_metrics_panel(
        self,
        frame: np.ndarray,
        analysis: AnalysisFrame,
        processing_fps: float,
        frame_index: int,
        total_frames: int,
    ) -> None:
        lines = [
            "Crowd Clustering Surveillance",
            f"People: {analysis.person_count}",
            f"Clusters: {analysis.cluster_count}",
            f"Stationary: {analysis.stationary_count}",
            f"FPS: {processing_fps:.1f}",
            f"Frame: {frame_index}/{total_frames or '?'}",
        ]

        panel_height = 28 + (len(lines) * 26)
        cv2.rectangle(frame, (10, 10), (320, panel_height), (18, 23, 33), -1)
        cv2.rectangle(frame, (10, 10), (320, panel_height), (56, 68, 92), 2)

        for index, line in enumerate(lines):
            color = (244, 247, 251) if index == 0 else (215, 223, 232)
            font_scale = 0.72 if index == 0 else 0.58
            thickness = 2 if index == 0 else 1
            cv2.putText(
                frame,
                line,
                (22, 34 + (index * 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
