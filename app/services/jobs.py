from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from typing import Deque
from uuid import uuid4

from app.config import MAX_HISTORY_POINTS, OUTPUT_DIR
from app.schemas import MetricsSnapshot, ProcessingConfig
from app.services.video_processor import CrowdVideoProcessor


@dataclass
class VideoJob:
    job_id: str
    filename: str
    input_path: Path
    output_path: Path
    config: ProcessingConfig
    status: str = "queued"
    error: str | None = None
    total_frames: int = 0
    frame_index: int = 0
    processing_fps: float = 0.0
    person_count: int = 0
    cluster_count: int = 0
    stationary_count: int = 0
    active_cluster_ids: list[int] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    latest_frame: bytes | None = None
    output_ready: bool = False
    output_path_str: str | None = None
    history_frames: Deque[float] = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_POINTS))
    history_people: Deque[float] = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_POINTS))
    history_clusters: Deque[float] = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_POINTS))
    history_stationary: Deque[float] = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_POINTS))
    history_fps: Deque[float] = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_POINTS))
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    pause_gate: threading.Event = field(default_factory=threading.Event, repr=False)

    def __post_init__(self) -> None:
        self.pause_gate.set()

    def mark_started(self, total_frames: int, output_path: str) -> None:
        with self.lock:
            self.status = "running"
            self.total_frames = total_frames
            self.output_path_str = output_path

    def wait_if_paused(self) -> None:
        while not self.pause_gate.is_set():
            time.sleep(0.1)

    def pause(self) -> None:
        with self.lock:
            if self.status == "running":
                self.status = "paused"
                self.pause_gate.clear()

    def resume(self) -> None:
        with self.lock:
            if self.status == "paused":
                self.status = "running"
                self.pause_gate.set()

    def update_frame(self, frame_index: int, total_frames: int, processing_fps: float, analysis, frame_bytes: bytes | None) -> None:
        with self.lock:
            self.frame_index = frame_index
            self.total_frames = total_frames
            self.processing_fps = processing_fps
            self.person_count = analysis.person_count
            self.cluster_count = analysis.cluster_count
            self.stationary_count = analysis.stationary_count
            self.active_cluster_ids = [cluster.cluster_id for cluster in analysis.clusters]
            if frame_bytes is not None:
                self.latest_frame = frame_bytes

            self.history_frames.append(frame_index)
            self.history_people.append(analysis.person_count)
            self.history_clusters.append(analysis.cluster_count)
            self.history_stationary.append(analysis.stationary_count)
            self.history_fps.append(round(processing_fps, 2))

    def mark_completed(self) -> None:
        with self.lock:
            self.status = "completed"
            self.output_ready = True
            self.pause_gate.set()

    def mark_failed(self, error: str) -> None:
        with self.lock:
            self.status = "failed"
            self.error = error
            self.pause_gate.set()

    def progress(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return min(self.frame_index / self.total_frames, 1.0)

    def snapshot(self) -> MetricsSnapshot:
        with self.lock:
            progress = self.progress()
            stream_url = f"/api/jobs/{self.job_id}/stream"
            download_url = f"/api/jobs/{self.job_id}/download" if self.output_ready else None
            history = {
                "frames": list(self.history_frames),
                "people": list(self.history_people),
                "clusters": list(self.history_clusters),
                "stationary": list(self.history_stationary),
                "fps": list(self.history_fps),
            }
            return MetricsSnapshot(
                job_id=self.job_id,
                filename=self.filename,
                status=self.status,
                person_count=self.person_count,
                cluster_count=self.cluster_count,
                stationary_count=self.stationary_count,
                processing_fps=self.processing_fps,
                frame_index=self.frame_index,
                total_frames=self.total_frames,
                progress=progress,
                export_progress=progress,
                active_cluster_ids=list(self.active_cluster_ids),
                history=history,
                error=self.error,
                download_url=download_url,
                stream_url=stream_url,
            )


class JobManager:
    def __init__(self, max_workers: int = 2) -> None:
        self.jobs: dict[str, VideoJob] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def create_job(self, input_path: Path, filename: str, config: ProcessingConfig) -> VideoJob:
        job_id = uuid4().hex
        safe_name = Path(filename).stem.replace(" ", "_") or "processed"
        output_path = OUTPUT_DIR / f"{job_id}_{safe_name}.mp4"
        job = VideoJob(
            job_id=job_id,
            filename=filename,
            input_path=input_path,
            output_path=output_path,
            config=config.sanitized(),
        )
        with self.lock:
            self.jobs[job_id] = job
        self.executor.submit(self._run_job, job)
        return job

    def get_job(self, job_id: str) -> VideoJob | None:
        with self.lock:
            return self.jobs.get(job_id)

    def pause_job(self, job_id: str) -> VideoJob:
        job = self._require_job(job_id)
        job.pause()
        return job

    def resume_job(self, job_id: str) -> VideoJob:
        job = self._require_job(job_id)
        job.resume()
        return job

    def stream_frames(self, job_id: str):
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        last_frame_index = -1

        while True:
            job = self._require_job(job_id)
            snapshot = job.snapshot()
            with job.lock:
                frame = job.latest_frame
                frame_index = job.frame_index
                status = job.status

            if frame is not None and frame_index != last_frame_index:
                last_frame_index = frame_index
                yield boundary + frame + b"\r\n"
            elif status in {"completed", "failed"} and frame is None:
                break

            if status in {"completed", "failed"} and frame_index == last_frame_index:
                time.sleep(0.3)
                if snapshot.status == "completed":
                    yield boundary + (frame or b"") + b"\r\n"
                break

            time.sleep(0.08)

    def _run_job(self, job: VideoJob) -> None:
        try:
            processor = CrowdVideoProcessor(job.config)
            processor.process(job.input_path, job.output_path, job)
        except Exception as exc:  # pragma: no cover - runtime protection
            job.mark_failed(str(exc))

    def _require_job(self, job_id: str) -> VideoJob:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return job
