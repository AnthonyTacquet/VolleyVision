from pathlib import Path

import cv2
import numpy as np
import yaml

from app.datasets import inspect_dataset
from app.pipeline import _apply_roi, _parse_yolo_labels


def test_inspect_standard_yolo_dataset(tmp_path: Path):
    root = tmp_path / "sample_export"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image_path = root / "images" / split / "frame001.jpg"
        assert cv2.imwrite(str(image_path), image)
        (root / "labels" / split / "frame001.txt").write_text("1 0.5 0.5 0.1 0.1\n")
    (root / "data.yaml").write_text(yaml.safe_dump({
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "player", 1: "ball"},
        "nc": 2,
    }))
    info = inspect_dataset(root / "data.yaml")
    assert info["images"] == 2
    assert info["labelled_images"] == 2
    assert info["suggested_class"] == 1
    assert info["counts"]["train"] == 1
    assert info["counts"]["val"] == 1


def test_ball_label_filter_remaps_to_zero(tmp_path: Path):
    labels = tmp_path / "labels.txt"
    labels.write_text("0 0.2 0.3 0.1 0.1\n1 0.5 0.5 0.2 0.2\n")
    assert _parse_yolo_labels(labels, 1) == ["0 0.5 0.5 0.2 0.2"]


def test_court_roi_keeps_court_and_greys_background(tmp_path: Path):
    court = tmp_path / "court.json"
    court.write_text('''{
      "frame_size": [100, 100],
      "landmarks": {
        "near_left": [10, 80], "near_right": [90, 80],
        "far_left": [30, 20], "far_right": [70, 20]
      }
    }''')
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    masked = _apply_roi(frame, str(court), margin=0.0)
    assert np.all(masked[50, 50] == 255)
    assert np.all(masked[5, 5] == 114)


def test_docker_remaps_host_absolute_yolo_paths(tmp_path: Path):
    root = tmp_path / "mounted" / "ball_yolo"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    image_path = root / "images" / "train" / "frame001.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))
    (root / "labels" / "train" / "frame001.txt").write_text("0 0.5 0.5 0.05 0.05\n")
    old_host_root = "/home/hostuser/project/export/dataset/ball_yolo"
    (root / "data.yaml").write_text(yaml.safe_dump({
        "path": old_host_root,
        "train": old_host_root + "/images/train",
        "names": ["ball"],
        "nc": 1,
    }))
    info = inspect_dataset(root / "data.yaml")
    assert info["counts"]["train"] == 1
    assert info["splits"]["train"][0]["image"] == str(image_path.resolve())


def test_court_roi_scales_original_calibration_resolution(tmp_path: Path):
    court = tmp_path / "court.json"
    court.write_text('''{
      "frame_size": [100, 100],
      "landmarks": {
        "near_left": [10, 80], "near_right": [90, 80],
        "far_left": [30, 20], "far_right": [70, 20]
      }
    }''')
    frame = np.full((50, 50, 3), 255, dtype=np.uint8)
    masked = _apply_roi(frame, str(court), margin=0.0)
    assert np.all(masked[25, 25] == 255)
    assert np.all(masked[2, 2] == 114)


def test_inspect_native_volley_json_point_annotations_and_events(tmp_path: Path):
    import json
    from app.datasets import inspect_volley_annotation, load_volley_ball_frames, load_volley_events

    labels_dir = tmp_path / "labels"
    videos_dir = tmp_path / "videos"
    labels_dir.mkdir()
    videos_dir.mkdir()
    video = videos_dir / "match.mp4"
    video.write_bytes(b"placeholder")
    annotation = labels_dir / "match.mp4.json"
    annotation.write_text(json.dumps({
        "video": "../old/path/match.mp4",
        "frame_size": [1280, 720],
        "ball": {
            "656": {"x": 435.1, "y": 338.4, "v": 1, "s": 16},
            "657": {"x": 434.0, "y": 333.0, "v": 1, "s": 16},
            "658": {"v": 0},
            "659": {"v": 0}
        },
        "events": [{"id": 1, "frame": 656, "type": "attack", "bbox": [1, 2, 3, 4]}],
        "meta": {"ball_size": 16}
    }), encoding="utf-8")

    info = inspect_volley_annotation(annotation, [videos_dir])
    assert info["source_type"] == "volley_json"
    assert info["video_path"] == str(video.resolve())
    assert info["ball_visible"] == 2
    assert info["ball_invisible"] == 2
    assert info["event_types"] == {"attack": 1}
    records = load_volley_ball_frames(annotation)
    assert len(records) == 4
    assert records[0]["frame_index"] == 656
    assert records[0]["visible"] is True
    assert records[0]["size"] == 16
    assert records[2]["visible"] is False
    events = load_volley_events(annotation)
    assert events[0]["type"] == "spike"
    assert events[0]["bbox"] == [1, 2, 3, 4]


def test_native_volley_ball_point_converts_to_square_yolo_box():
    from app.pipeline import _ball_yolo_line

    line = _ball_yolo_line({"visible": True, "x": 50, "y": 25, "size": 16}, 100, 50)
    parts = line.split()
    assert parts[0] == "0"
    assert float(parts[1]) == 0.5
    assert float(parts[2]) == 0.5
    assert abs(float(parts[3]) - 0.16) < 1e-6
    assert abs(float(parts[4]) - 0.32) < 1e-6
    assert _ball_yolo_line({"visible": False}, 100, 50) is None


def test_event_aliases_match_labeler_vocabulary():
    from app.datasets import normalise_event_type

    assert normalise_event_type("attack") == "spike"
    assert normalise_event_type("bounce") == "floor"
    assert normalise_event_type("reception") == "reception"


def test_event_annotations_convert_to_yolo_boxes():
    from app.pipeline import _event_yolo_lines

    lines = _event_yolo_lines([
        {"type": "attack", "bbox": [10, 5, 30, 25]},
        {"type": "spike", "point": [50, 25]},
        {"type": "floor"},
    ], 100, 50, "spike")
    assert len(lines) == 2
    assert all(line.startswith("0 ") for line in lines)
    assert _event_yolo_lines([{"type": "bounce"}], 100, 50, "floor")[0] == "0 0.50000000 0.50000000 1.00000000 1.00000000"


def test_native_volley_json_discovery_avoids_export_duplicate(tmp_path: Path):
    import json
    from app.datasets import discover_sources

    root = tmp_path / "data"
    labels = root / "labels"
    export_labels = root / "export" / "dataset_x" / "labels"
    videos = root / "videos"
    labels.mkdir(parents=True)
    export_labels.mkdir(parents=True)
    videos.mkdir(parents=True)
    (videos / "match.mp4").write_bytes(b"placeholder")
    payload = {"video": "match.mp4", "frame_size": [100, 100], "ball": {"0": {"x": 20, "y": 20, "v": 1, "s": 12}}, "events": []}
    (labels / "match.mp4.json").write_text(json.dumps(payload), encoding="utf-8")
    (export_labels / "match.mp4.json").write_text(json.dumps(payload), encoding="utf-8")

    datasets, warnings = discover_sources([root], [videos])
    native = [d for d in datasets if d.get("source_type") == "volley_json"]
    assert len(native) == 1
    assert native[0]["video_available"] is True
    assert native[0]["ball_visible"] == 1


def test_event_only_annotation_is_discoverable_and_collectable(tmp_path: Path):
    import json
    from app.datasets import inspect_volley_annotation
    from app.pipeline import _collect_items

    labels = tmp_path / "labels"
    videos = tmp_path / "videos"
    labels.mkdir()
    videos.mkdir()
    video = videos / "match.mp4"
    video.write_bytes(b"placeholder")
    annotation = labels / "match.mp4.json"
    annotation.write_text(json.dumps({
        "video": "match.mp4",
        "frame_size": [100, 50],
        "ball": {},
        "events": [
            {"id": 1, "frame": 12, "type": "serve", "point": [40, 20]},
            {"id": 2, "frame": 13, "type": "spike", "point": [50, 20]},
            {"id": 3, "frame": 14, "type": "dig", "point": [60, 20]},
        ],
    }), encoding="utf-8")

    dataset = inspect_volley_annotation(annotation, [videos])
    assert dataset["images"] == 3
    train, _validation = _collect_items(
        [{"dataset": dataset, "class_id": 0, "calibration": None}],
        "group_by_source", 42, lambda _message: None, "serve",
    )
    assert train[0]["events"][0]["type"] == "serve"
    assert train[1]["events"][0]["type"] == "spike"
    from app.pipeline import _event_yolo_lines
    assert _event_yolo_lines(train[1]["events"], 100, 50, "serve") == []


def test_annotation_json_video_fallback_uses_annotation_filename(tmp_path: Path):
    import json
    from app.datasets import inspect_volley_annotation

    labels = tmp_path / "labels"
    videos = tmp_path / "videos"
    labels.mkdir()
    videos.mkdir()
    video = videos / "clip_sample-match.mp4"
    video.write_bytes(b"placeholder")
    annotation = labels / "clip_sample-match.mp4.json"
    annotation.write_text(json.dumps({
        "video": "../sample-match/clips/clip.mp4",
        "frame_size": [1280, 720],
        "ball": {"12": {"x": 100, "y": 200, "v": 1, "s": 16}},
        "events": []
    }), encoding="utf-8")

    info = inspect_volley_annotation(annotation, [videos])
    assert info["video_path"] == str(video.resolve())


def test_write_split_extracts_native_video_frames_and_negative_labels(tmp_path: Path):
    from app.pipeline import _write_split

    video = tmp_path / "match.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    if not writer.isOpened():
        import pytest
        pytest.skip("OpenCV MP4 writer codec unavailable in this environment")
    for i in range(3):
        writer.write(np.full((48, 64, 3), i * 60, dtype=np.uint8))
    writer.release()

    items = [
        {"source_type": "volley_json", "source_id": "abc123456789", "video_path": str(video), "frame_index": 0,
         "visible": True, "x": 32, "y": 24, "size": 12, "frame_size": [64, 48]},
        {"source_type": "volley_json", "source_id": "abc123456789", "video_path": str(video), "frame_index": 1,
         "visible": False, "frame_size": [64, 48]},
    ]
    destination = tmp_path / "prepared"
    count = _write_split(items, "train", destination, 0.35, False, lambda _message: None)
    assert count == 2
    labels = sorted((destination / "labels" / "train").glob("*.txt"))
    assert len(labels) == 2
    contents = [p.read_text().strip() for p in labels]
    assert any(line.startswith("0 ") for line in contents)
    assert "" in contents
    images = list((destination / "images" / "train").glob("*.jpg"))
    assert len(images) == 2
