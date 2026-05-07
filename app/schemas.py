from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProcessingConfig:
    model_name: str = "yolov8n.pt"
    device: str | None = None
    imgsz: int = 640
    conf_thresh: float = 0.25
    iou_thresh: float = 0.45
    stationarity_seconds: float = 6.0
    stationarity_radius: float = 70.0
    cluster_eps: float = 115.0
    min_cluster_size: int = 3
    history_seconds: float = 12.0
    perspective_height_reference: float = 170.0
    perspective_gain: float = 0.85
    trail_length: int = 24
    track_buffer: int = 45
    track_high_thresh: float = 0.4
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.45
    match_thresh: float = 0.8

    def sanitized(self) -> "ProcessingConfig":
        device = (self.device or "").strip() or None
        history_seconds = max(float(self.history_seconds), float(self.stationarity_seconds) + 2.0)
        return ProcessingConfig(
            model_name=(self.model_name or "yolov8n.pt").strip() or "yolov8n.pt",
            device=device,
            imgsz=max(320, min(1280, int(self.imgsz))),
            conf_thresh=max(0.05, min(0.95, float(self.conf_thresh))),
            iou_thresh=max(0.05, min(0.95, float(self.iou_thresh))),
            stationarity_seconds=max(2.0, min(20.0, float(self.stationarity_seconds))),
            stationarity_radius=max(20.0, min(250.0, float(self.stationarity_radius))),
            cluster_eps=max(30.0, min(350.0, float(self.cluster_eps))),
            min_cluster_size=max(3, int(self.min_cluster_size)),
            history_seconds=history_seconds,
            perspective_height_reference=max(60.0, min(320.0, float(self.perspective_height_reference))),
            perspective_gain=max(0.0, min(2.0, float(self.perspective_gain))),
            trail_length=max(4, min(80, int(self.trail_length))),
            track_buffer=max(10, min(180, int(self.track_buffer))),
            track_high_thresh=max(0.05, min(0.95, float(self.track_high_thresh))),
            track_low_thresh=max(0.01, min(0.9, float(self.track_low_thresh))),
            new_track_thresh=max(0.05, min(0.95, float(self.new_track_thresh))),
            match_thresh=max(0.1, min(0.99, float(self.match_thresh))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "imgsz": self.imgsz,
            "conf_thresh": self.conf_thresh,
            "iou_thresh": self.iou_thresh,
            "stationarity_seconds": self.stationarity_seconds,
            "stationarity_radius": self.stationarity_radius,
            "cluster_eps": self.cluster_eps,
            "min_cluster_size": self.min_cluster_size,
            "history_seconds": self.history_seconds,
            "perspective_height_reference": self.perspective_height_reference,
            "perspective_gain": self.perspective_gain,
            "trail_length": self.trail_length,
            "track_buffer": self.track_buffer,
            "track_high_thresh": self.track_high_thresh,
            "track_low_thresh": self.track_low_thresh,
            "new_track_thresh": self.new_track_thresh,
            "match_thresh": self.match_thresh,
        }


@dataclass
class MetricsSnapshot:
    job_id: str
    filename: str
    status: str = "queued"
    person_count: int = 0
    cluster_count: int = 0
    stationary_count: int = 0
    processing_fps: float = 0.0
    frame_index: int = 0
    total_frames: int = 0
    progress: float = 0.0
    export_progress: float = 0.0
    active_cluster_ids: list[int] = field(default_factory=list)
    history: dict[str, list[float]] = field(default_factory=dict)
    error: str | None = None
    download_url: str | None = None
    stream_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status,
            "person_count": self.person_count,
            "cluster_count": self.cluster_count,
            "stationary_count": self.stationary_count,
            "processing_fps": round(float(self.processing_fps), 2),
            "frame_index": self.frame_index,
            "total_frames": self.total_frames,
            "progress": round(float(self.progress), 4),
            "export_progress": round(float(self.export_progress), 4),
            "active_cluster_ids": self.active_cluster_ids,
            "history": self.history,
            "error": self.error,
            "download_url": self.download_url,
            "stream_url": self.stream_url,
        }
