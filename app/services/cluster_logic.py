from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque

import numpy as np
from sklearn.cluster import DBSCAN

from app.schemas import ProcessingConfig


@dataclass
class Observation:
    frame_index: int
    timestamp: float
    bbox: tuple[int, int, int, int]
    pixel_anchor: np.ndarray
    world_anchor: np.ndarray


@dataclass
class TrackHistory:
    track_id: int
    observations: Deque[Observation]
    last_seen_frame: int = 0

    def add(self, observation: Observation) -> None:
        self.observations.append(observation)
        self.last_seen_frame = observation.frame_index


@dataclass
class PersonState:
    track_id: int
    bbox: tuple[int, int, int, int]
    pixel_anchor: tuple[int, int]
    world_anchor: tuple[float, float]
    speed: float
    stationary_seconds: float
    is_stationary: bool
    trail: list[tuple[int, int]]


@dataclass
class ClusterState:
    cluster_id: int
    member_track_ids: list[int]
    centroid_pixel: tuple[int, int]
    centroid_world: tuple[float, float]
    radius_px: int


@dataclass
class AnalysisFrame:
    persons: list[PersonState] = field(default_factory=list)
    clusters: list[ClusterState] = field(default_factory=list)

    @property
    def person_count(self) -> int:
        return len(self.persons)

    @property
    def cluster_count(self) -> int:
        return len(self.clusters)

    @property
    def stationary_count(self) -> int:
        return sum(1 for person in self.persons if person.is_stationary)


@dataclass
class ClusterMemory:
    cluster_id: int
    centroid_world: np.ndarray
    last_seen_frame: int


class PerspectiveNormalizer:
    def __init__(self, config: ProcessingConfig) -> None:
        self.config = config

    def to_world(self, bbox: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        frame_h, frame_w = frame_shape[:2]
        foot_x = (x1 + x2) / 2.0
        foot_y = float(y2)
        height = max(float(y2 - y1), 1.0)

        depth_from_height = self.config.perspective_height_reference / height
        vertical_position = min(max(foot_y / max(frame_h, 1), 0.0), 1.0)
        row_gain = 1.0 + (1.0 - vertical_position) * self.config.perspective_gain
        scale = min(max(depth_from_height * row_gain, 0.6), 4.5)

        return np.array(
            [
                (foot_x - (frame_w / 2.0)) * scale,
                foot_y * scale,
            ],
            dtype=np.float32,
        )


class CrowdClusterAnalyzer:
    def __init__(self, config: ProcessingConfig, fps: float) -> None:
        self.config = config
        self.fps = max(float(fps), 1.0)
        self.normalizer = PerspectiveNormalizer(config)
        history_size = max(int(config.history_seconds * self.fps * 1.5), config.trail_length * 3)
        self.history_size = min(max(history_size, 64), 720)
        self.track_histories: dict[int, TrackHistory] = {}
        self.cluster_memories: dict[int, ClusterMemory] = {}
        self.next_cluster_id = 1

    def analyze(self, tracks: np.ndarray, frame_index: int, frame_shape: tuple[int, ...]) -> AnalysisFrame:
        current_track_ids: set[int] = set()
        persons: list[PersonState] = []

        for track in tracks:
            x1, y1, x2, y2 = [int(value) for value in track[:4]]
            track_id = int(track[4])
            bbox = (x1, y1, x2, y2)
            current_track_ids.add(track_id)

            pixel_anchor = np.array([(x1 + x2) / 2.0, float(y2)], dtype=np.float32)
            world_anchor = self.normalizer.to_world(bbox, frame_shape)
            observation = Observation(
                frame_index=frame_index,
                timestamp=frame_index / self.fps,
                bbox=bbox,
                pixel_anchor=pixel_anchor,
                world_anchor=world_anchor,
            )
            history = self.track_histories.setdefault(
                track_id,
                TrackHistory(track_id=track_id, observations=deque(maxlen=self.history_size)),
            )
            history.add(observation)

            speed, stationary_seconds, is_stationary = self._stationary_metrics(history)
            trail = self._trail_points(history)
            persons.append(
                PersonState(
                    track_id=track_id,
                    bbox=bbox,
                    pixel_anchor=(int(pixel_anchor[0]), int(pixel_anchor[1])),
                    world_anchor=(float(world_anchor[0]), float(world_anchor[1])),
                    speed=speed,
                    stationary_seconds=stationary_seconds,
                    is_stationary=is_stationary,
                    trail=trail,
                )
            )

        self._cleanup_tracks(frame_index, current_track_ids)
        clusters = self._build_clusters(persons, frame_index)
        self._cleanup_clusters(frame_index)

        return AnalysisFrame(persons=persons, clusters=clusters)

    def _stationary_metrics(self, history: TrackHistory) -> tuple[float, float, bool]:
        recent = list(history.observations)
        if len(recent) < 2:
            return 0.0, 0.0, False

        cutoff = recent[-1].timestamp - self.config.stationarity_seconds
        window = [obs for obs in recent if obs.timestamp >= cutoff]
        if len(window) < 3:
            return 0.0, 0.0, False

        duration = max(window[-1].timestamp - window[0].timestamp, 0.0)
        if duration + (1.0 / self.fps) < self.config.stationarity_seconds:
            return duration, 0.0, False

        world_points = np.array([obs.world_anchor for obs in window], dtype=np.float32)
        center = np.median(world_points, axis=0)
        offsets = world_points - center
        radial_distances = np.sqrt((offsets * offsets).sum(axis=1))
        max_radius = float(radial_distances.max()) if len(radial_distances) else 0.0

        if len(world_points) > 1:
            deltas = np.diff(world_points, axis=0)
            path_length = float(np.sqrt((deltas * deltas).sum(axis=1)).sum())
        else:
            path_length = 0.0

        speed = path_length / max(duration, 1e-6)
        stationary_speed_cap = self.config.stationarity_radius / self.config.stationarity_seconds
        is_stationary = (
            max_radius <= self.config.stationarity_radius
            and path_length <= self.config.stationarity_radius * 2.5
            and speed <= stationary_speed_cap
        )
        stationary_seconds = duration if is_stationary else 0.0

        return speed, stationary_seconds, is_stationary

    def _trail_points(self, history: TrackHistory) -> list[tuple[int, int]]:
        points = [obs.pixel_anchor for obs in list(history.observations)[-self.config.trail_length :]]
        return [(int(point[0]), int(point[1])) for point in points]

    def _build_clusters(self, persons: list[PersonState], frame_index: int) -> list[ClusterState]:
        stationary_people = [person for person in persons if person.is_stationary]
        if len(stationary_people) < self.config.min_cluster_size:
            return []

        coordinates = np.array([person.world_anchor for person in stationary_people], dtype=np.float32)
        labels = DBSCAN(eps=self.config.cluster_eps, min_samples=self.config.min_cluster_size).fit_predict(coordinates)

        clusters: list[ClusterState] = []
        for label in sorted(set(labels)):
            if label == -1:
                continue

            members = [stationary_people[index] for index, item in enumerate(labels) if item == label]
            if len(members) < self.config.min_cluster_size:
                continue

            pixel_points = np.array([person.pixel_anchor for person in members], dtype=np.float32)
            world_points = np.array([person.world_anchor for person in members], dtype=np.float32)

            centroid_pixel = pixel_points.mean(axis=0)
            centroid_world = world_points.mean(axis=0)
            pixel_offsets = pixel_points - centroid_pixel
            if len(pixel_offsets):
                max_radius = float(np.sqrt((pixel_offsets * pixel_offsets).sum(axis=1)).max())
            else:
                max_radius = 0.0
            radius_px = max(36, int(math.ceil(max_radius + 32.0)))
            cluster_id = self._assign_cluster_id(centroid_world, frame_index)

            clusters.append(
                ClusterState(
                    cluster_id=cluster_id,
                    member_track_ids=[person.track_id for person in members],
                    centroid_pixel=(int(centroid_pixel[0]), int(centroid_pixel[1])),
                    centroid_world=(float(centroid_world[0]), float(centroid_world[1])),
                    radius_px=radius_px,
                )
            )

        return clusters

    def _assign_cluster_id(self, centroid_world: np.ndarray, frame_index: int) -> int:
        match_id: int | None = None
        best_distance = float("inf")
        stale_frame_limit = frame_index - int(self.fps * 2.0)

        for cluster_id, memory in self.cluster_memories.items():
            if memory.last_seen_frame < stale_frame_limit:
                continue
            distance = float(np.linalg.norm(centroid_world - memory.centroid_world))
            if distance < best_distance and distance <= self.config.cluster_eps * 1.5:
                best_distance = distance
                match_id = cluster_id

        if match_id is None:
            match_id = self.next_cluster_id
            self.next_cluster_id += 1

        self.cluster_memories[match_id] = ClusterMemory(
            cluster_id=match_id,
            centroid_world=np.array(centroid_world, dtype=np.float32),
            last_seen_frame=frame_index,
        )
        return match_id

    def _cleanup_tracks(self, frame_index: int, current_track_ids: set[int]) -> None:
        stale_frame_limit = frame_index - int(self.fps * self.config.history_seconds)
        removable = [
            track_id
            for track_id, history in self.track_histories.items()
            if history.last_seen_frame < stale_frame_limit and track_id not in current_track_ids
        ]
        for track_id in removable:
            self.track_histories.pop(track_id, None)

    def _cleanup_clusters(self, frame_index: int) -> None:
        stale_frame_limit = frame_index - int(self.fps * 2.0)
        removable = [
            cluster_id
            for cluster_id, memory in self.cluster_memories.items()
            if memory.last_seen_frame < stale_frame_limit
        ]
        for cluster_id in removable:
            self.cluster_memories.pop(cluster_id, None)
