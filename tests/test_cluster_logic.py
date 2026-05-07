from __future__ import annotations

import numpy as np

from app.schemas import ProcessingConfig
from app.services.cluster_logic import CrowdClusterAnalyzer


def _track_row(x1: int, y1: int, x2: int, y2: int, track_id: int) -> np.ndarray:
    return np.array([x1, y1, x2, y2, track_id, 0.9, 0.0, 0.0], dtype=np.float32)


def test_stationary_people_form_cluster_after_persistent_duration() -> None:
    config = ProcessingConfig(
        stationarity_seconds=3.0,
        stationarity_radius=35.0,
        cluster_eps=120.0,
        min_cluster_size=3,
        history_seconds=6.0,
    ).sanitized()
    analyzer = CrowdClusterAnalyzer(config=config, fps=10.0)
    frame_shape = (720, 1280, 3)

    for frame_index in range(1, 40):
        tracks = np.array(
            [
                _track_row(100, 220, 150, 360, 1),
                _track_row(155, 225, 205, 365, 2),
                _track_row(205, 230, 255, 370, 3),
            ]
        )
        analysis = analyzer.analyze(tracks, frame_index=frame_index, frame_shape=frame_shape)

    assert analysis.stationary_count == 3
    assert analysis.cluster_count == 1
    assert sorted(analysis.clusters[0].member_track_ids) == [1, 2, 3]


def test_moving_group_is_not_classified_as_cluster() -> None:
    config = ProcessingConfig(
        stationarity_seconds=3.0,
        stationarity_radius=35.0,
        cluster_eps=120.0,
        min_cluster_size=3,
        history_seconds=6.0,
    ).sanitized()
    analyzer = CrowdClusterAnalyzer(config=config, fps=10.0)
    frame_shape = (720, 1280, 3)

    for frame_index in range(1, 40):
        offset = frame_index * 12
        tracks = np.array(
            [
                _track_row(100 + offset, 220, 150 + offset, 360, 1),
                _track_row(155 + offset, 225, 205 + offset, 365, 2),
                _track_row(205 + offset, 230, 255 + offset, 370, 3),
            ]
        )
        analysis = analyzer.analyze(tracks, frame_index=frame_index, frame_shape=frame_shape)

    assert analysis.stationary_count == 0
    assert analysis.cluster_count == 0


def test_perspective_normalization_expands_far_object_motion() -> None:
    config = ProcessingConfig().sanitized()
    analyzer = CrowdClusterAnalyzer(config=config, fps=25.0)
    frame_shape = (1080, 1920, 3)

    near_world = analyzer.normalizer.to_world((800, 600, 920, 980), frame_shape)
    far_world = analyzer.normalizer.to_world((820, 150, 875, 320), frame_shape)

    near_step = analyzer.normalizer.to_world((810, 600, 930, 980), frame_shape) - near_world
    far_step = analyzer.normalizer.to_world((830, 150, 885, 320), frame_shape) - far_world

    assert np.linalg.norm(far_step) > np.linalg.norm(near_step)
