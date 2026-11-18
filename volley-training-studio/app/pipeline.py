from __future__ import annotations

import csv
import json
import random
import shutil
import sys
import traceback
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml

from .datasets import EVENT_TARGETS, discover_sources, load_volley_ball_frames, load_volley_events, normalise_event_type

LogFn = Callable[[str], None]


def _select_compute_device(cuda_available: bool) -> int | str:
    return 0 if cuda_available else "cpu"


def _load_court_polygon(court_json: str, width: int, height: int, margin: float) -> np.ndarray:
    data = json.loads(Path(court_json).read_text(encoding="utf-8"))
    landmarks = data.get("landmarks", {})
    names = ("near_left", "near_right", "far_right", "far_left")
    if not all(name in landmarks for name in names):
        raise ValueError(f"Court calibration {court_json} is missing one of {names}")
    points = np.array([landmarks[name] for name in names], dtype=np.float32)
    reference_size = data.get("frame_size")
    if isinstance(reference_size, (list, tuple)) and len(reference_size) == 2:
        reference_width, reference_height = float(reference_size[0]), float(reference_size[1])
        if reference_width > 0 and reference_height > 0:
            points[:, 0] *= width / reference_width
            points[:, 1] *= height / reference_height
    center = points.mean(axis=0)
    points = center + (points - center) * (1.0 + max(0.0, margin))
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return np.round(points).astype(np.int32)


def _apply_roi(image: np.ndarray, court_json: str, margin: float) -> np.ndarray:
    h, w = image.shape[:2]
    polygon = _load_court_polygon(court_json, w, h, margin)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    background = np.full_like(image, 114)
    return np.where(mask[:, :, None] > 0, image, background)


def _parse_yolo_labels(label_path: Path, class_id: int) -> list[str]:
    kept: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw_line.strip().split()
        if not parts:
            continue
        try:
            source_class = int(float(parts[0]))
        except ValueError:
            continue
        if source_class != class_id:
            continue
        if len(parts) != 5:
            raise ValueError(
                f"Expected YOLO detection labels with 5 columns in {label_path}; "
                "segmentation labels are not supported by this ball detector pipeline."
            )
        kept.append("0 " + " ".join(parts[1:]))
    return kept


def _read_source(source_id: str, discovered: list[dict[str, Any]]) -> dict[str, Any]:
    for item in discovered:
        if item["id"] == source_id:
            return item
    raise ValueError(f"Dataset source {source_id} is no longer available. Refresh the dataset list.")


def _temporal_split(items: list[dict[str, Any]], log: LogFn, source_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(items, key=lambda x: int(x.get("frame_index", 0)))
    if len(ordered) < 2:
        return ordered, []
    cut = max(1, min(len(ordered) - 1, int(len(ordered) * 0.8)))
    log(f"Temporal split for {source_name}: first {cut} annotated frames for training, final {len(ordered) - cut} for validation.")
    return ordered[:cut], ordered[cut:]


def _collect_items(selected: list[dict[str, Any]], split_mode: str, seed: int, log: LogFn, target: str = "ball_tracking") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_items: list[dict[str, Any]] = []
    per_source: dict[str, list[dict[str, Any]]] = {}
    for src in selected:
        dataset = src["dataset"]
        source_id = dataset["id"]
        source_name = dataset["name"]
        source_type = dataset.get("source_type", "yolo")
        source_items: list[dict[str, Any]] = []
        if source_type == "volley_json":
            annotation_path = Path(dataset["annotation_path"])
            if target == "ball_tracking":
                records = load_volley_ball_frames(annotation_path)
            else:
                events = load_volley_events(annotation_path)
                event_frames: dict[int, list[dict[str, Any]]] = {}
                for event in events:
                    event_frames.setdefault(event["frame_index"], []).append(event)
                try:
                    frame_indices = {record["frame_index"] for record in load_volley_ball_frames(annotation_path)}
                except ValueError:
                    frame_indices = set()
                frame_indices.update(event_frames)
                records = [{"frame_index": frame_index, "events": event_frames.get(frame_index, [])}
                           for frame_index in sorted(frame_indices)]
            for record in records:
                source_items.append({
                    **record,
                    "source_type": "volley_json",
                    "source_id": source_id,
                    "source_name": source_name,
                    "video_path": dataset["video_path"],
                    "frame_size": dataset.get("frame_size"),
                    "calibration": src.get("calibration"),
                    "declared_split": "",
                })
            if target == "ball_tracking":
                visible_count = sum(1 for item in source_items if item.get("visible"))
                negative_count = len(source_items) - visible_count
                log(f"Loaded VolleyVision JSON {source_name}: {len(source_items)} annotated frames, {visible_count} visible ball points, {negative_count} explicit invisible frames, {dataset.get('event_count', 0)} action events.")
            else:
                event_count = sum(len(item.get("events", [])) for item in source_items)
                log(f"Loaded VolleyVision JSON {source_name}: {len(source_items)} annotated frames and {event_count} event annotations.")
        else:
            for split_name in ("train", "val", "test"):
                for entry in dataset["splits"].get(split_name, []):
                    if not entry.get("label"):
                        continue
                    source_items.append({
                        **entry,
                        "source_type": "yolo",
                        "source_id": source_id,
                        "source_name": source_name,
                        "class_id": int(src["class_id"]),
                        "calibration": src.get("calibration"),
                        "declared_split": split_name,
                    })
        per_source[source_id] = source_items
        all_items.extend(source_items)

    if split_mode == "group_by_source" and len(selected) >= 2:
        source_ids = [x["dataset"]["id"] for x in selected]
        rng = random.Random(seed)
        rng.shuffle(source_ids)
        val_count = max(1, round(len(source_ids) * 0.2))
        val_sources = set(source_ids[:val_count])
        train_items = [x for x in all_items if x["source_id"] not in val_sources]
        val_items = [x for x in all_items if x["source_id"] in val_sources]
        log(f"Group split: {len(train_items)} training frames from {len(source_ids) - val_count} source(s); {len(val_items)} validation frames from {val_count} held-out source(s).")
        return train_items, val_items

    if len(selected) == 1 and selected[0]["dataset"].get("source_type") == "volley_json":
        return _temporal_split(all_items, log, selected[0]["dataset"]["name"])

    if split_mode == "use_yaml_splits":
        train_items: list[dict[str, Any]] = []
        val_items: list[dict[str, Any]] = []
        for src in selected:
            items = per_source[src["dataset"]["id"]]
            if src["dataset"].get("source_type") == "volley_json":
                train, val = _temporal_split(items, log, src["dataset"]["name"])
                train_items.extend(train)
                val_items.extend(val)
            else:
                train_items.extend(x for x in items if x["declared_split"] == "train")
                val_items.extend(x for x in items if x["declared_split"] == "val")
        if train_items and val_items:
            return train_items, val_items

    # For a single YOLO source use its declared train/val split when available.
    train_items = [x for x in all_items if x.get("declared_split") == "train"]
    val_items = [x for x in all_items if x.get("declared_split") == "val"]
    if train_items and val_items:
        return train_items, val_items

    # Fallback keeps a contiguous temporal holdout for native JSON sources and a
    # deterministic image split for YOLO-only exports.
    json_items = [x for x in all_items if x.get("source_type") == "volley_json"]
    yolo_items = [x for x in all_items if x.get("source_type") == "yolo"]
    if json_items and not yolo_items:
        return _temporal_split(json_items, log, "selected VolleyVision clips")
    rng = random.Random(seed)
    rng.shuffle(all_items)
    cut = max(1, min(len(all_items) - 1, int(len(all_items) * 0.8))) if len(all_items) > 1 else len(all_items)
    log("Warning: generated a deterministic 80/20 image split; adjacent video frames may be correlated.")
    return all_items[:cut], all_items[cut:]


def _ball_yolo_line(item: dict[str, Any], width: int, height: int, box_scale: float = 1.0) -> str | None:
    if not item.get("visible"):
        return None
    x = float(item["x"])
    y = float(item["y"])
    side = float(item.get("size", 16.0)) * max(0.25, float(box_scale))
    reference_size = item.get("frame_size")
    if isinstance(reference_size, (list, tuple)) and len(reference_size) == 2:
        ref_w, ref_h = float(reference_size[0]), float(reference_size[1])
        if ref_w > 0 and ref_h > 0:
            sx, sy = width / ref_w, height / ref_h
            x *= sx
            y *= sy
            side *= (sx + sy) / 2.0
    half = side / 2.0
    x1 = max(0.0, x - half)
    y1 = max(0.0, y - half)
    x2 = min(float(width), x + half)
    y2 = min(float(height), y + half)
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    nw, nh = box_w / width, box_h / height
    return f"0 {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}"


def _event_yolo_lines(events: list[dict[str, Any]], width: int, height: int, target: str) -> list[str]:
    lines = []
    for event in events:
        if normalise_event_type(event.get("type")) != target:
            continue
        reference_size = event.get("frame_size")
        scale_x = scale_y = 1.0
        if isinstance(reference_size, (list, tuple)) and len(reference_size) == 2:
            reference_width, reference_height = float(reference_size[0]), float(reference_size[1])
            if reference_width > 0 and reference_height > 0:
                scale_x, scale_y = width / reference_width, height / reference_height
        bbox = event.get("bbox")
        point = event.get("point")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = (float(value) for value in bbox)
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
            y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
        elif isinstance(point, (list, tuple)) and len(point) == 2:
            x, y = float(point[0]) * scale_x, float(point[1]) * scale_y
            side = max(8.0, min(width, height) * 0.04)
            x1, x2 = max(0.0, x - side / 2), min(float(width), x + side / 2)
            y1, y2 = max(0.0, y - side / 2), min(float(height), y + side / 2)
        else:
            x1, y1, x2, y2 = 0.0, 0.0, float(width), float(height)
        box_width, box_height = x2 - x1, y2 - y1
        if box_width <= 0 or box_height <= 0:
            continue
        center_x, center_y = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        lines.append(f"0 {center_x:.8f} {center_y:.8f} {box_width / width:.8f} {box_height / height:.8f}")
    return lines


def _write_split(items: list[dict[str, Any]], split: str, destination: Path, margin: float, mask_background: bool, log: LogFn, ball_box_scale: float = 1.0, target: str = "ball_tracking") -> int:
    image_dir = destination / "images" / split
    label_dir = destination / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    seen: set[str] = set()
    captures: dict[str, cv2.VideoCapture] = {}
    try:
        for index, item in enumerate(items):
            frame = None
            label_lines: list[str] = []
            if item.get("source_type") == "volley_json":
                video_path = str(item["video_path"])
                cap = captures.get(video_path)
                if cap is None:
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        log(f"Warning: could not open source video; skipped {video_path}")
                        continue
                    captures[video_path] = cap
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(item["frame_index"]))
                ok, frame = cap.read()
                if not ok or frame is None:
                    log(f"Warning: could not decode frame {item['frame_index']} from {video_path}")
                    continue
                if target == "ball_tracking":
                    line = _ball_yolo_line(item, frame.shape[1], frame.shape[0], ball_box_scale)
                    if line:
                        label_lines.append(line)
                else:
                    label_lines = _event_yolo_lines(item.get("events", []), frame.shape[1], frame.shape[0], target)
            else:
                image_path = Path(item["image"])
                label_path = Path(item["label"])
                if not image_path.is_file() or not label_path.is_file():
                    continue
                label_lines = _parse_yolo_labels(label_path, int(item["class_id"]))
                frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if frame is None:
                    log(f"Warning: could not decode image; skipped {image_path}")
                    continue

            if mask_background and item.get("calibration"):
                frame = _apply_roi(frame, item["calibration"], margin)
            source_key = item["source_id"][:8]
            frame_suffix = f"_{int(item['frame_index']):08d}" if item.get("source_type") == "volley_json" else ""
            filename = f"{source_key}_{index:07d}{frame_suffix}.jpg"
            while filename in seen:
                filename = f"{source_key}_{index:07d}_{len(seen)}{frame_suffix}.jpg"
            seen.add(filename)
            output_image = image_dir / filename
            if not cv2.imwrite(str(output_image), frame):
                log(f"Warning: could not write {output_image}; skipped")
                continue
            # An empty label file is intentional for v:0 / not-visible frames.
            (label_dir / f"{Path(filename).stem}.txt").write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            written += 1
    finally:
        for cap in captures.values():
            cap.release()
    return written


def _write_tracking_script(path: Path) -> None:
    script = r'''#!/usr/bin/env python3
"""Track the trained volleyball detector through a video.

Example:
  python track_ball.py --model best.pt --source match.mp4 --court-json court.json
  python track_ball.py --model best.pt --source match.mp4 --tracker ocsort.yaml --no-mask
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def roi_mask(frame, court_json, margin):
    data = json.loads(Path(court_json).read_text(encoding="utf-8"))
    lm = data["landmarks"]
    names = ("near_left", "near_right", "far_right", "far_left")
    pts = np.array([lm[n] for n in names], dtype=np.float32)
    reference_size = data.get("frame_size")
    h, w = frame.shape[:2]
    if isinstance(reference_size, (list, tuple)) and len(reference_size) == 2:
        reference_width, reference_height = float(reference_size[0]), float(reference_size[1])
        if reference_width > 0 and reference_height > 0:
            pts[:, 0] *= w / reference_width
            pts[:, 1] *= h / reference_height
    center = pts.mean(axis=0)
    pts = center + (pts - center) * (1.0 + max(0.0, margin))
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    return np.where(mask[:, :, None] > 0, frame, np.full_like(frame, 114))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Trained best.pt")
    parser.add_argument("--source", required=True, help="Input video")
    parser.add_argument("--tracker", default="ocsort.yaml")
    parser.add_argument("--conf", type=float, default=0.03)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--court-json", help="Optional court.json to apply the same expanded ROI mask")
    parser.add_argument("--margin", type=float, default=0.35, help="ROI expansion fraction around calibrated court")
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--output", help="Output MP4 path")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.output) if args.output else source.with_name(source.stem + "_tracked.mp4")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix(".csv")
    model = YOLO(args.model)
    frame_no = 0
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        rows = csv.writer(handle)
        rows.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "confidence"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            processed = frame
            if not args.no_mask and args.court_json:
                processed = roi_mask(frame, args.court_json, args.margin)
            result = model.track(processed, persist=True, tracker=args.tracker,
                                 classes=[0], conf=args.conf, imgsz=args.imgsz,
                                 verbose=False)[0]
            annotated = result.plot()
            writer.write(annotated)
            boxes = result.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(xyxy)
                for box, track_id, confidence in zip(xyxy, ids, confs):
                    rows.writerow([frame_no, int(track_id), *[round(float(v), 2) for v in box], round(float(confidence), 5)])
            frame_no += 1
    cap.release()
    writer.release()
    print(f"Saved tracked video: {out}")
    print(f"Saved track positions: {csv_path}")


if __name__ == "__main__":
    main()
'''
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def run_pipeline(
    config: dict[str, Any],
    job_dir: Path,
    output_root: Path,
    log: LogFn,
    set_status: Callable[[str, str | None], None],
    cancel_event: Any | None = None,
) -> dict[str, Any]:
    job_dir.mkdir(parents=True, exist_ok=True)
    target = str(config.get("target", "ball_tracking"))
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Training cancelled by user.")
    if target != "ball_tracking" and target not in EVENT_TARGETS:
        raise ValueError(f"Unsupported training target: {target}")
    log("Stage 1/4 — validating selected datasets…")
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Training cancelled by user.")
    source_roots = [Path(p) for p in config["source_roots"]]
    video_root = Path(config.get("video_root", "/videos"))
    datasets, warnings = discover_sources(source_roots, [video_root])
    for warning in warnings:
        log("Warning: " + warning)
    selected: list[dict[str, Any]] = []
    for selection in config["sources"]:
        dataset = next((d for d in datasets if d["id"] == selection["dataset_id"]), None)
        if dataset is None:
            raise ValueError(f"Selected source no longer exists: {selection['dataset_id']}")
        source_type = dataset.get("source_type", "yolo")
        class_id = int(selection.get("class_id", 0))
        if target != "ball_tracking" and source_type != "volley_json":
            raise ValueError(f"Event training requires Volley Labeler JSON annotations: {dataset['name']}")
        if source_type == "yolo":
            class_ids = {c["id"] for c in dataset["classes"]}
            if class_id not in class_ids:
                raise ValueError(f"Class {class_id} is not present in {dataset['name']}")
            class_name = next((c["name"] for c in dataset["classes"] if c["id"] == class_id), str(class_id))
        else:
            if not dataset.get("video_path") or not Path(dataset["video_path"]).is_file():
                raise ValueError(f"Source video is missing for {dataset['name']}: {dataset.get('video_path')}")
            class_name = "ball point annotations converted to square boxes" if target == "ball_tracking" else f"{target} event annotations"
        selected.append({"dataset": dataset, "class_id": class_id, "calibration": selection.get("calibration_path")})
        log(f"Selected {dataset['name']}: {dataset['images']} annotated frames, target '{class_name}'.")

    mask_background = bool(config.get("mask_background"))
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Training cancelled by user.")
    if mask_background:
        missing = [x["dataset"]["name"] for x in selected if not x.get("calibration")]
        if missing:
            raise ValueError("Court ROI background filtering is enabled, but no court.json was selected for: " + ", ".join(missing))
        log(f"Stage 2/4 — applying expanded court ROI mask (margin {config['margin']:.0%}) to training images…")
    else:
        log("Stage 2/4 — background filtering disabled; keeping original full frames.")

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Training cancelled by user.")
    train_items, val_items = _collect_items(selected, config.get("split_mode", "group_by_source"), int(config.get("seed", 42)), log, target)
    if target != "ball_tracking" and not any(
        normalise_event_type(event.get("type")) == target
        for item in train_items
        for event in item.get("events", [])
    ):
        raise ValueError(f"No '{target}' event annotations are present in the training split. Select another source or split seed.")
    if not train_items:
        raise ValueError("No labelled training frames found. Check that selected JSON files contain ball points or YOLO exports contain label files.")
    if not val_items:
        raise ValueError("No validation images found. Select at least two source datasets or ensure each data.yaml has a val split.")

    dataset_dir = job_dir / "dataset"
    ball_box_scale = float(config.get("ball_box_scale", 1.0))
    train_count = _write_split(train_items, "train", dataset_dir, float(config.get("margin", 0.35)), mask_background, log, ball_box_scale, target)
    val_count = _write_split(val_items, "val", dataset_dir, float(config.get("margin", 0.35)), mask_background, log, ball_box_scale, target)
    if not train_count or not val_count:
        raise ValueError(f"Dataset preparation produced train={train_count}, val={val_count}. Need both splits to be non-empty.")

    data_yaml = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["ball" if target == "ball_tracking" else target],
    }
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    (job_dir / "dataset_manifest.json").write_text(json.dumps({
        "sources": [{"name": s["dataset"]["name"], "source_type": s["dataset"].get("source_type", "yolo"), "yaml_path": s["dataset"].get("yaml_path"), "annotation_path": s["dataset"].get("annotation_path"), "video_path": s["dataset"].get("video_path"), "class_id": s["class_id"], "calibration": s.get("calibration")} for s in selected],
        "train_images": train_count, "val_images": val_count,
        "mask_background": mask_background, "mask_margin": config.get("margin", 0.35),
        "ball_box_scale": ball_box_scale,
        "split_mode": config.get("split_mode"), "seed": config.get("seed", 42)
    }, indent=2), encoding="utf-8")
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Training cancelled by user.")
    log(f"Prepared single-class {target} dataset: {train_count} train images, {val_count} validation images.")

    log("Stage 3/4 — starting YOLO training…")
    set_status("training", "training")
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Training cancelled by user.")
    import torch
    from ultralytics import YOLO

    device = _select_compute_device(torch.cuda.is_available())
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    log(f"Compute device: {device_name}")
    model_name = str(config.get("model", "yolo11n.pt"))
    run_dir = job_dir / "train_run"
    model = YOLO(model_name)

    def report_epoch(trainer: Any) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Training cancelled by user.")
        metrics = getattr(trainer, "metrics", {}) or {}
        metric_summary = []
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                try:
                    metric_summary.append(f"{key}={float(value):.4f}")
                except (TypeError, ValueError):
                    continue
        epoch_number = int(getattr(trainer, "epoch", 0)) + 1
        total_epochs = int(getattr(trainer, "epochs", config.get("epochs", 0)))
        suffix = " | " + ", ".join(metric_summary[:5]) if metric_summary else ""
        log(f"Epoch {epoch_number}/{total_epochs} complete{suffix}")

    # Keep an epoch-level progress trail in the website's job log without
    # redirecting process-wide stdout while the FastAPI server is running.
    model.add_callback("on_fit_epoch_end", report_epoch)
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Training cancelled by user.")
    model.train(
        data=str(yaml_path),
        epochs=int(config.get("epochs", 100)),
        imgsz=int(config.get("imgsz", 640)),
        batch=int(config.get("batch", 2)),
        device=device,
        workers=int(config.get("workers", 4)),
        project=str(job_dir),
        name="train_run",
        exist_ok=True,
        patience=int(config.get("patience", 25)),
        cache=False,
        plots=True,
        amp=device != "cpu",
        verbose=True,
    )
    best_path = run_dir / "weights" / "best.pt"
    if not best_path.is_file():
        raise RuntimeError("Training finished but best.pt was not found in the YOLO run directory.")

    log(f"Stage 4/4 — packaging {target} best.pt…")
    models_dir = output_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_output = models_dir / f"{job_dir.name}_{target}_best.pt"
    shutil.copy2(best_path, model_output)
    tracking_script = None
    if target == "ball_tracking":
        tracking_script = job_dir / "track_ball.py"
        _write_tracking_script(tracking_script)
        shutil.copy2(tracking_script, models_dir / f"{job_dir.name}_track_ball.py")

    csv_metrics = run_dir / "results.csv"
    summary = {
        "job_id": job_dir.name,
        "model": model_name,
        "weights": str(model_output),
        "train_images": train_count,
        "val_images": val_count,
        "gpu": device_name if device == 0 else None,
        "device": device_name,
        "results_dir": str(run_dir),
        "metrics_csv": str(csv_metrics) if csv_metrics.exists() else None,
    }
    (job_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    bundle_path = models_dir / f"{job_dir.name}_{target}_model_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(model_output, arcname="best.pt")
        if tracking_script:
            archive.write(models_dir / f"{job_dir.name}_track_ball.py", arcname="track_ball.py")
        archive.write(yaml_path, arcname="data.yaml")
        archive.write(job_dir / "dataset_manifest.json", arcname="dataset_manifest.json")
        archive.write(job_dir / "training_summary.json", arcname="training_summary.json")
        if csv_metrics.exists():
            archive.write(csv_metrics, arcname="results.csv")
    log(f"Training complete. Best weights: {model_output}")
    if tracking_script:
        log(f"Tracking helper: {models_dir / f'{job_dir.name}_track_ball.py'}")
    return {"model_path": str(model_output), "target": target, "tracking_script": str(models_dir / f"{job_dir.name}_track_ball.py") if tracking_script else None, "bundle_path": str(bundle_path), "train_images": train_count, "val_images": val_count, "gpu": device_name if device == 0 else None, "device": device_name}
