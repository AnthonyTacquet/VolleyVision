from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
DATA_YAML_NAMES = {"data.yaml", "data.yml"}
IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "runs", ".pytest_cache", "training", "output", "out", "videos", "calibrations", "models"}
EVENT_TARGETS = {"serve", "reception", "dig", "set", "spike", "block", "floor"}


def normalise_event_type(value: Any) -> str:
    lowered = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    if lowered in {"attack", "spike / attack", "spike/attack"}:
        return "spike"
    if lowered in {"bounce", "ball hits floor", "ball hit floor"}:
        return "floor"
    return lowered


def _normalise_names(value: Any) -> dict[int, str]:
    if isinstance(value, list):
        return {i: str(name) for i, name in enumerate(value)}
    if isinstance(value, dict):
        result: dict[int, str] = {}
        for key, name in value.items():
            try:
                result[int(key)] = str(name)
            except (TypeError, ValueError):
                continue
        return result
    return {}


def _resolve_path(raw: str | Path, base: Path, dataset_root: Path) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        parts = candidate.parts
        for index, part in enumerate(parts):
            if part.lower() in {"images", "labels"}:
                remapped = dataset_root / Path(*parts[index:])
                if remapped.exists():
                    return remapped
        remapped = dataset_root / candidate.name
        if remapped.exists():
            return remapped
        return candidate
    first = dataset_root / candidate
    if first.exists():
        return first
    return base / candidate


def _images_for_split(value: Any, base: Path, dataset_root: Path) -> list[Path]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    found: list[Path] = []
    for item in values:
        if not isinstance(item, (str, Path)):
            continue
        path = _resolve_path(str(item), base, dataset_root)
        if path.is_dir():
            found.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                listed = Path(line)
                if not listed.is_absolute():
                    listed = dataset_root / listed
                    if not listed.exists():
                        listed = path.parent / line
                if listed.is_file() and listed.suffix.lower() in IMAGE_SUFFIXES:
                    found.append(listed)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            found.append(path)
    result: list[Path] = []
    seen: set[str] = set()
    for p in sorted(found, key=lambda x: str(x)):
        resolved = str(p.resolve())
        if resolved not in seen:
            seen.add(resolved)
            result.append(p.resolve())
    return result


def _label_path(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            candidate = Path(*parts).with_suffix(".txt")
            return candidate if candidate.is_file() else None
    candidate = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    if candidate.is_file():
        return candidate
    candidate = image_path.with_suffix(".txt")
    return candidate if candidate.is_file() else None


def inspect_dataset(yaml_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    base = yaml_path.parent.resolve()
    raw_root = raw.get("path")
    if raw_root:
        root_candidate = Path(str(raw_root)).expanduser()
        if root_candidate.is_absolute():
            dataset_root = root_candidate if root_candidate.exists() else base
        else:
            dataset_root = base / root_candidate
    else:
        dataset_root = base
    dataset_root = dataset_root.resolve()
    names = _normalise_names(raw.get("names"))

    splits: dict[str, list[dict[str, str | None]]] = {}
    total = 0
    labelled = 0
    for split_name in ("train", "val", "test"):
        images = _images_for_split(raw.get(split_name), base, dataset_root)
        entries: list[dict[str, str | None]] = []
        for image in images:
            label = _label_path(image)
            entries.append({"image": str(image), "label": str(label) if label else None})
            total += 1
            labelled += int(label is not None)
        splits[split_name] = entries

    digest = hashlib.sha1(str(yaml_path.resolve()).encode()).hexdigest()[:12]
    suggested_class = None
    for class_id, name in names.items():
        lowered = name.lower().replace("_", " ").replace("-", " ")
        if "ball" in lowered or "volleyball" in lowered:
            suggested_class = class_id
            break
    if suggested_class is None and len(names) == 1:
        suggested_class = next(iter(names))

    return {
        "id": digest,
        "source_type": "yolo",
        "name": yaml_path.parent.name,
        "yaml_path": str(yaml_path.resolve()),
        "root": str(dataset_root),
        "classes": [{"id": key, "name": value} for key, value in sorted(names.items())],
        "suggested_class": suggested_class,
        "counts": {name: len(items) for name, items in splits.items()},
        "images": total,
        "labelled_images": labelled,
        "splits": splits,
    }


def _find_ball_frame_map(value: Any, path: str = "", depth: int = 0) -> tuple[dict[str, Any] | None, str | None]:
    """Find a frame-indexed map whose values look like {x, y, v, s} ball points."""
    if depth > 6 or not isinstance(value, dict):
        return None, None
    numeric_items = [(k, v) for k, v in value.items() if str(k).isdigit() and isinstance(v, dict)]
    if numeric_items:
        point_like = [v for _, v in numeric_items if ("x" in v and "y" in v) or "v" in v or "visible" in v]
        # Require a meaningful fraction of the numeric records to look like point annotations.
        if point_like and len(point_like) / max(1, len(numeric_items)) >= 0.5:
            return value, path or "<root>"
    # Prefer dictionaries reached through a key containing "ball".
    ordered = sorted(value.items(), key=lambda kv: (0 if "ball" in str(kv[0]).lower() else 1, str(kv[0])))
    for key, child in ordered:
        found, found_path = _find_ball_frame_map(child, f"{path}.{key}" if path else str(key), depth + 1)
        if found is not None:
            return found, found_path
    return None, None


def _resolve_video(raw_video: Any, annotation_path: Path, video_roots: list[Path]) -> Path | None:
    raw_text = str(raw_video).strip() if raw_video else ""
    candidates: list[Path] = []
    if raw_text:
        raw_path = Path(raw_text).expanduser()
        candidates.extend([raw_path, annotation_path.parent / raw_path])
        for root in video_roots:
            candidates.append(root / raw_path.name)
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_SUFFIXES:
            return candidate.resolve()

    # The stored annotation may have been renamed when copied into data/labels,
    # while its original `video` field still refers to e.g. .../clips/clip.mp4.
    # Try both that basename and the JSON filename (without its trailing .json).
    names: list[str] = []
    if raw_text:
        names.append(Path(raw_text).name)
    names.append(annotation_path.name.removesuffix(".json"))
    for basename in dict.fromkeys(name for name in names if name):
        matches: list[Path] = []
        for root in video_roots:
            if not root.exists():
                continue
            direct = root / basename
            if direct.is_file():
                return direct.resolve()
            matches.extend(p for p in root.rglob(basename) if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
            if matches:
                break
        if matches:
            return sorted(matches, key=lambda p: str(p))[0].resolve()
        stem = Path(basename).stem
        if stem:
            for root in video_roots:
                if not root.exists():
                    continue
                fuzzy = [p for p in root.rglob(f"{stem}.*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
                if fuzzy:
                    return sorted(fuzzy, key=lambda p: str(p))[0].resolve()
    return None


def inspect_volley_annotation(annotation_path: Path, video_roots: list[Path]) -> dict[str, Any]:
    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Annotation root must be a JSON object")
    frame_map, frame_map_path = _find_ball_frame_map(raw)
    events = raw.get("events") if isinstance(raw.get("events"), list) else []
    if frame_map is None:
        event_frames = []
        for event in events:
            if isinstance(event, dict):
                try:
                    event_frames.append(int(event["frame"]))
                except (KeyError, TypeError, ValueError):
                    continue
        if not event_frames:
            raise ValueError("Could not find frame-indexed ball points or valid event annotations")
        frame_map = {}

    visible = invisible = invalid = 0
    frame_indices: list[int] = []
    for key, point in frame_map.items():
        if not str(key).isdigit() or not isinstance(point, dict):
            continue
        frame_indices.append(int(key))
        v = point.get("v", point.get("visible", 1))
        is_visible = bool(v) and str(v).lower() not in {"0", "false", "none"}
        if is_visible and "x" in point and "y" in point:
            try:
                float(point["x"]); float(point["y"])
                visible += 1
            except (TypeError, ValueError):
                invalid += 1
        else:
            invisible += 1
    frame_size = raw.get("frame_size")
    event_types: dict[str, int] = {}
    for event in events:
        if isinstance(event, dict):
            event_type = str(event.get("type", "unknown"))
            event_types[event_type] = event_types.get(event_type, 0) + 1
    annotated_frames = set(frame_indices)
    for event in events:
        if isinstance(event, dict):
            try:
                annotated_frames.add(int(event["frame"]))
            except (KeyError, TypeError, ValueError):
                continue
    video_path = _resolve_video(raw.get("video"), annotation_path, video_roots)
    digest = hashlib.sha1(str(annotation_path.resolve()).encode()).hexdigest()[:12]
    return {
        "id": digest,
        "source_type": "volley_json",
        "name": annotation_path.name.removesuffix(".json"),
        "annotation_path": str(annotation_path.resolve()),
        "video_reference": raw.get("video"),
        "video_path": str(video_path) if video_path else None,
        "video_available": video_path is not None,
        "ball_map_path": frame_map_path,
        "frame_size": frame_size,
        "counts": {"annotated": len(annotated_frames), "visible": visible, "invisible": invisible},
        "images": len(annotated_frames),
        "labelled_images": len(annotated_frames),
        "ball_visible": visible,
        "ball_invisible": invisible,
        "invalid_points": invalid,
        "event_count": len(events),
        "event_types": event_types,
        "classes": [{"id": 0, "name": "ball (from point labels)"}],
        "suggested_class": 0,
        "splits": {},
    }


def _annotation_candidates(source_roots: list[Path]):
    for root in source_roots:
        if not root.exists():
            continue
        if root.name.lower() == "labels":
            search_roots = [root]
        elif (root / "labels").is_dir():
            # The user's data root contains the canonical labels/ folder as well as
            # an exported labels/ copy. Use only the canonical folder to avoid
            # training twice on the same annotation JSON.
            search_roots = [root / "labels"]
        else:
            search_roots = [root]
        for search_root in search_roots:
            for path in search_root.rglob("*.mp4.json"):
                if any(part in IGNORED_DIRS for part in path.parts):
                    continue
                if path.name.endswith(".bak") or not path.is_file():
                    continue
                yield path


def discover_datasets(source_roots: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    datasets: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for root in source_roots:
        if not root.exists():
            warnings.append(f"Dataset source directory is not mounted or does not exist: {root}")
            continue
        for current, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            for filename in files:
                if filename.lower() not in DATA_YAML_NAMES:
                    continue
                yaml_path = Path(current) / filename
                try:
                    item = inspect_dataset(yaml_path)
                    if yaml_path.parent.name.lower() in {"ball_yolo", "yolo", "dataset", "data"} and yaml_path.parent.parent != root:
                        item["name"] = f"{yaml_path.parent.parent.name} / {yaml_path.parent.name}"
                    if item["id"] in seen:
                        continue
                    seen.add(item["id"])
                    if item["images"] == 0:
                        warnings.append(f"No images found for {yaml_path}")
                    datasets.append(item)
                except Exception as exc:
                    warnings.append(f"Could not read {yaml_path}: {exc}")
    datasets.sort(key=lambda x: x["yaml_path"].lower())
    return datasets, warnings


def discover_sources(source_roots: list[Path], video_roots: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    datasets, warnings = discover_datasets(source_roots)
    seen_annotations: set[tuple[str, str]] = set()
    for annotation_path in _annotation_candidates(source_roots):
        try:
            item = inspect_volley_annotation(annotation_path, video_roots)
            # Deduplicate exact annotation copies by video and normalized content source.
            identity = (Path(item["video_path"]).name if item.get("video_path") else annotation_path.stem, item["source_type"])
            if identity in seen_annotations:
                continue
            seen_annotations.add(identity)
            if not item["video_available"]:
                warnings.append(f"No matching source video found for {annotation_path.name}; check VIDEO_HOST_PATH.")
            if item["ball_visible"] == 0 and item["ball_invisible"] > 0:
                warnings.append(f"No visible ball points found in {annotation_path.name}.")
            datasets.append(item)
        except Exception as exc:
            warnings.append(f"Could not read VolleyVision labels {annotation_path}: {exc}")
    datasets.sort(key=lambda x: (x.get("source_type", ""), x.get("name", "").lower()))
    return datasets, warnings


def discover_calibrations(root: Path) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if not root.exists():
        return result
    for path in root.rglob("court.json"):
        result.append({"id": hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12],
                       "name": str(path.relative_to(root)), "path": str(path.resolve()),
                       "video_stem": path.parent.name})
    return sorted(result, key=lambda x: x["name"].lower())


def load_volley_ball_frames(annotation_path: Path) -> list[dict[str, Any]]:
    """Return frame-indexed ball annotations as center points and visibility flags.

    The current Volley Labeler stores ball data as a mapping keyed by frame
    number. `s` is treated as the square bounding-box side length in source pixels.
    Invisible `v: 0` frames are retained as negative detector examples.
    """
    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    frame_map, _ = _find_ball_frame_map(raw)
    if frame_map is None:
        raise ValueError(f"No frame-indexed ball annotations found in {annotation_path}")
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    default_size = float(meta.get("ball_size", 16) or 16)
    result: list[dict[str, Any]] = []
    for key, point in frame_map.items():
        if not str(key).isdigit() or not isinstance(point, dict):
            continue
        frame_index = int(key)
        v = point.get("v", point.get("visible", 1))
        visible = bool(v) and str(v).lower() not in {"0", "false", "none"}
        entry: dict[str, Any] = {"frame_index": frame_index, "visible": visible}
        if visible:
            if "x" not in point or "y" not in point:
                continue
            try:
                entry["x"] = float(point["x"])
                entry["y"] = float(point["y"])
                entry["size"] = float(point.get("s", default_size) or default_size)
            except (TypeError, ValueError):
                continue
            if entry["size"] <= 0:
                entry["size"] = default_size
        result.append(entry)
    return sorted(result, key=lambda x: x["frame_index"])


def load_volley_events(annotation_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    frame_size = raw.get("frame_size") if isinstance(raw, dict) else None
    events = raw.get("events", []) if isinstance(raw, dict) else []
    result = []
    for event in events:
        if not isinstance(event, dict):
            continue
        try:
            frame_index = int(event["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        event_type = normalise_event_type(event.get("type"))
        if event_type not in EVENT_TARGETS:
            continue
        result.append({"frame_index": frame_index, "type": event_type,
                       "bbox": event.get("bbox"), "point": event.get("point"),
                       "frame_size": frame_size})
    return result
