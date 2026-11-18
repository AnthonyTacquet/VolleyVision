from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
MODEL_TARGETS = ("ball_tracking", "serve", "reception", "dig", "set", "spike", "block", "floor")
MODEL_TARGET_LABELS = {
    "ball_tracking": "Ball tracking",
    "serve": "Serve",
    "reception": "Reception",
    "dig": "Dig",
    "set": "Set",
    "spike": "Spike",
    "block": "Block",
    "floor": "Ball hits floor",
}
LABEL_TYPES = ("all", "ball", "events")
PORTAL_DIR = Path(__file__).resolve().parent


class RenameRequest(BaseModel):
    new_name: str = Field(min_length=1, max_length=120)


def _validate_segment(value: str) -> str:
    value = value.strip()
    if (
        not value
        or value in {".", ".."}
        or len(value) > 120
        or "/" in value
        or "\\" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise HTTPException(status_code=400, detail="Enter a valid name without path separators.")
    return value


def _child(root: Path, name: str) -> Path:
    if Path(name).name != name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    root = root.resolve()
    path = root / name
    if path.is_symlink() or path.resolve().parent != root:
        raise HTTPException(status_code=400, detail="Invalid file path.")
    return path


def _model_id(root_index: int, relative_path: str) -> str:
    value = f"{root_index}:{relative_path}"
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _model_path(model_id: str, model_roots: list[Path]) -> tuple[Path, int]:
    try:
        padding = "=" * (-len(model_id) % 4)
        value = base64.b64decode(model_id + padding, altchars=b"-_", validate=True).decode("utf-8")
        root_text, relative = value.split(":", 1)
        root_index = int(root_text)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid model identifier.") from exc
    relative_path = PurePosixPath(relative)
    if (
        root_index < 0
        or root_index >= len(model_roots)
        or relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.suffix.lower() != ".pt"
        or (root_index == 0 and len(relative_path.parts) != 1)
    ):
        raise HTTPException(status_code=400, detail="Invalid model path.")
    root = model_roots[root_index].resolve()
    candidate = root / Path(*relative_path.parts)
    if candidate.is_symlink():
        raise HTTPException(status_code=400, detail="Invalid model path.")
    path = candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Model path is outside the model library.") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Model file not found.")
    return path, root_index


def _model_task(path: Path, root_index: int) -> tuple[str, str] | None:
    for target in MODEL_TARGETS:
        suffix = f"_{target}_best.pt"
        if path.name.endswith(suffix):
            return target, suffix
    if path.name.endswith("_ball_best.pt"):
        return "ball_tracking", "_ball_best.pt"
    if root_index > 0 and path.suffix.lower() == ".pt":
        return "ball_tracking", ".pt"
    return None


def _label_video_path(videos_root: Path, name: str) -> Path:
    if not name or "/" in name or "\\" in name or any(ord(char) < 32 or ord(char) == 127 for char in name):
        raise HTTPException(status_code=400, detail="Invalid video name.")
    path = _child(videos_root, name)
    if path.suffix.lower() not in VIDEO_EXTENSIONS or not path.is_file():
        raise HTTPException(status_code=404, detail="Video not found.")
    return path


def _read_label_doc(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=409, detail="The video's label file could not be read safely.") from exc
    if not isinstance(doc, dict):
        raise HTTPException(status_code=409, detail="The video's label file has an invalid format.")
    if not isinstance(doc.get("ball", {}), dict) or not isinstance(doc.get("events", []), list):
        raise HTTPException(status_code=409, detail="The video's label file has invalid ball or event labels.")
    doc.setdefault("ball", {})
    doc.setdefault("events", [])
    return doc


def _write_label_doc(path: Path, doc: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".label-import-", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(doc, temporary)
        if path.exists():
            backup = path.with_suffix(".json.bak")
            if not backup.exists() or path.stat().st_mtime - backup.stat().st_mtime > 600:
                shutil.copy2(path, backup)
        os.replace(temporary_path, path)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not save label file: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def create_data_manager_app(
    videos_dir: Path,
    labels_dir: Path,
    model_roots: list[Path],
    video_cache: dict[str, Any],
    cache_lock: threading.Lock,
) -> FastAPI:
    videos_root = Path(videos_dir).resolve()
    labels_root = Path(labels_dir).resolve()
    resolved_model_roots = [Path(root).resolve() for root in model_roots]
    app = FastAPI(title="VolleyVision Data Manager")
    app.mount("/static", StaticFiles(directory=PORTAL_DIR / "static"), name="static")

    @app.get("/", response_class=FileResponse)
    def home() -> FileResponse:
        return FileResponse(PORTAL_DIR / "static" / "data.html")

    @app.get("/api/videos")
    def list_videos(query: str = Query(default="", max_length=120)) -> dict[str, list[dict[str, Any]]]:
        search = query.strip().casefold()
        videos = []
        if videos_root.exists():
            for path in sorted(videos_root.iterdir(), key=lambda item: item.name.lower()):
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.suffix.lower() not in VIDEO_EXTENSIONS
                    or (search and search not in path.name.casefold())
                ):
                    continue
                videos.append({"name": path.name, "stem": path.stem, "size_bytes": path.stat().st_size})
        return {"videos": videos}

    @app.get("/api/labels")
    def list_labels(
        query: str = Query(default="", max_length=120),
        type: str = Query(default="all"),
    ) -> dict[str, list[dict[str, Any]]]:
        if type not in LABEL_TYPES:
            raise HTTPException(status_code=400, detail="Select all, ball, or event labels.")
        search = query.strip().casefold()
        labels = []
        if videos_root.exists() and labels_root.exists():
            for video in sorted(videos_root.iterdir(), key=lambda item: item.name.lower()):
                if (
                    video.is_symlink()
                    or not video.is_file()
                    or video.suffix.lower() not in VIDEO_EXTENSIONS
                    or (search and search not in video.name.casefold())
                ):
                    continue
                label_path = _child(labels_root, f"{video.name}.json")
                if label_path.is_symlink() or not label_path.is_file():
                    continue
                doc = _read_label_doc(label_path)
                labels.append({
                    "video": video.name,
                    "ball_labels": len(doc.get("ball", {})),
                    "event_labels": len(doc.get("events", [])),
                    "size_bytes": label_path.stat().st_size,
                })
        if type == "ball":
            labels = [label for label in labels if label["ball_labels"]]
        elif type == "events":
            labels = [label for label in labels if label["event_labels"]]
        return {"labels": labels}

    @app.post("/api/labels/import")
    def import_labels(target: str = Form(...), file: UploadFile = File(...)) -> dict[str, Any]:
        if target not in LABEL_TYPES:
            raise HTTPException(status_code=400, detail="Select all, ball, or event labels before importing.")
        original_name = _validate_segment(file.filename or "")
        if Path(original_name).suffix.lower() != ".json":
            raise HTTPException(status_code=400, detail="Import a VolleyVision label file with the .json extension.")
        payload = file.file.read(10 * 1024 * 1024 + 1)
        if len(payload) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="The label file is larger than 10 MB.")
        try:
            imported = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="The selected file is not valid UTF-8 JSON.") from exc
        if not isinstance(imported, dict) or not isinstance(imported.get("video"), str):
            raise HTTPException(status_code=400, detail="The label file must include its associated video name.")
        video_name = imported["video"]
        _label_video_path(videos_root, video_name)
        if not isinstance(imported.get("ball", {}), dict) or not isinstance(imported.get("events", []), list):
            raise HTTPException(status_code=400, detail="The label file has invalid ball or event labels.")
        if target == "ball" and not isinstance(imported.get("ball"), dict):
            raise HTTPException(status_code=400, detail="The label file does not contain valid ball labels.")
        if target == "events" and not isinstance(imported.get("events"), list):
            raise HTTPException(status_code=400, detail="The label file does not contain valid event labels.")

        destination = _child(labels_root, f"{video_name}.json")
        with cache_lock:
            existing = _read_label_doc(destination) if destination.exists() else {}
            try:
                revision = int(existing.get("rev", 0)) + 1
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=409, detail="The video's label revision is invalid.") from exc
            if target == "all":
                doc = imported
            else:
                doc = existing
                doc[target] = imported[target]
            doc["video"] = video_name
            doc.setdefault("ball", {})
            doc.setdefault("events", [])
            doc["rev"] = revision
            _write_label_doc(destination, doc)
        return {
            "video": video_name,
            "ball_labels": len(doc.get("ball", {})),
            "event_labels": len(doc.get("events", [])),
        }

    @app.get("/api/labels/{name}/download")
    def download_labels(name: str, type: str = Query(default="all")) -> JSONResponse:
        if type not in LABEL_TYPES:
            raise HTTPException(status_code=400, detail="Select all, ball, or event labels.")
        video = _label_video_path(videos_root, name)
        path = _child(labels_root, f"{video.name}.json")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Label file not found.")
        doc = _read_label_doc(path)
        if type == "ball":
            doc["events"] = []
        elif type == "events":
            doc["ball"] = {}
        filename = quote(f"{video.name}.json", safe="")
        return JSONResponse(
            content=doc,
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
        )

    @app.delete("/api/labels/{name}")
    def delete_labels(name: str, type: str = Query(default="all")) -> dict[str, str]:
        if type not in LABEL_TYPES:
            raise HTTPException(status_code=400, detail="Select all, ball, or event labels.")
        video = _label_video_path(videos_root, name)
        path = _child(labels_root, f"{video.name}.json")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Label file not found.")
        with cache_lock:
            if type == "all":
                try:
                    path.unlink()
                    _child(labels_root, f"{video.name}.json.bak").unlink(missing_ok=True)
                except OSError as exc:
                    raise HTTPException(status_code=500, detail=f"Could not delete label file: {exc}") from exc
            else:
                doc = _read_label_doc(path)
                doc[type] = {} if type == "ball" else []
                if not doc["ball"] and not doc["events"]:
                    try:
                        path.unlink()
                    except OSError as exc:
                        raise HTTPException(status_code=500, detail=f"Could not delete label file: {exc}") from exc
                else:
                    _write_label_doc(path, doc)
        return {"video": video.name, "deleted": type}

    @app.get("/api/models")
    def list_models() -> dict[str, list[dict[str, Any]]]:
        found = []
        for root_index, root in enumerate(resolved_model_roots):
            if not root.is_dir():
                continue
            candidates = root.glob("*_best.pt") if root_index == 0 else root.rglob("*.pt")
            for path in candidates:
                if path.is_symlink() or not path.is_file():
                    continue
                task_info = _model_task(path, root_index)
                if task_info is None:
                    continue
                task, suffix = task_info
                try:
                    stat = path.stat()
                    relative = path.relative_to(root).as_posix()
                except (OSError, ValueError):
                    continue
                found.append({
                    "id": _model_id(root_index, relative),
                    "name": path.name,
                    "edit_name": path.name[:-len(suffix)],
                    "task": task,
                    "task_label": MODEL_TARGET_LABELS[task],
                    "source": "Training workspace" if root_index == 0 else f"Read-only model library {root_index}",
                    "size_bytes": stat.st_size,
                    "can_rename": root_index == 0,
                })
        found.sort(key=lambda model: (model["task_label"].casefold(), model["name"].casefold()))
        return {"models": found}

    @app.get("/api/videos/{name}/download")
    def download_video(name: str) -> FileResponse:
        path = _child(videos_root, name)
        if path.suffix.lower() not in VIDEO_EXTENSIONS or not path.is_file():
            raise HTTPException(status_code=404, detail="Video not found.")
        return FileResponse(path, filename=path.name)

    @app.get("/api/models/{model_id}/download")
    def download_model(model_id: str) -> FileResponse:
        path, _root_index = _model_path(model_id, resolved_model_roots)
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")

    @app.post("/api/models/import")
    def import_model(target: str = Form(...), file: UploadFile = File(...)) -> dict[str, Any]:
        if target not in MODEL_TARGETS:
            raise HTTPException(status_code=400, detail="Select a specific model type before importing.")

        original_name = _validate_segment(file.filename or "")
        if Path(original_name).suffix.lower() != ".pt":
            raise HTTPException(status_code=400, detail="Import a PyTorch model file with the .pt extension.")

        stem = Path(original_name).stem
        known_suffixes = [f"_{item}_best" for item in MODEL_TARGETS] + ["_ball_best"]
        for suffix in known_suffixes:
            if stem.lower().endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        stem = _validate_segment(stem)
        if not resolved_model_roots:
            raise HTTPException(status_code=503, detail="No writable model folder is configured.")

        models_root = resolved_model_roots[0]
        destination = _child(models_root, f"{stem}_{target}_best.pt")
        temporary_path: Path | None = None
        try:
            models_root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".model-import-", suffix=".tmp", dir=models_root, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                size_bytes = 0
                while chunk := file.file.read(1024 * 1024):
                    temporary.write(chunk)
                    size_bytes += len(chunk)
            if size_bytes == 0:
                raise HTTPException(status_code=400, detail="The selected model file is empty.")
            try:
                os.link(temporary_path, destination)
            except FileExistsError as exc:
                raise HTTPException(status_code=409, detail="A model with that name already exists.") from exc
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Could not import model: {exc}") from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not import model: {exc}") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return {
            "name": destination.name,
            "task": target,
            "task_label": MODEL_TARGET_LABELS[target],
            "size_bytes": size_bytes,
        }

    @app.post("/api/videos/{name}/rename")
    def rename_video(name: str, request: RenameRequest) -> dict[str, str]:
        source = _child(videos_root, name)
        if source.suffix.lower() not in VIDEO_EXTENSIONS or not source.is_file():
            raise HTTPException(status_code=404, detail="Video not found.")
        stem = _validate_segment(request.new_name)
        if stem.lower().endswith(source.suffix.lower()):
            stem = stem[: -len(source.suffix)]
        if not stem or stem in {".", ".."}:
            raise HTTPException(status_code=400, detail="Enter a valid video name.")
        destination = _child(videos_root, f"{stem}{source.suffix}")
        if destination == source:
            return {"name": source.name}
        if destination.exists():
            raise HTTPException(status_code=409, detail="A video with that name already exists.")

        old_label = _child(labels_root, f"{source.name}.json")
        new_label = _child(labels_root, f"{destination.name}.json")
        old_backup = _child(labels_root, f"{source.name}.json.bak")
        new_backup = _child(labels_root, f"{destination.name}.json.bak")
        label_doc = None
        original_label = None
        if new_label.exists():
            raise HTTPException(status_code=409, detail="A label file with that name already exists.")
        if new_backup.exists():
            raise HTTPException(status_code=409, detail="A label backup with that name already exists.")
        if old_label.exists():
            try:
                original_label = old_label.read_text(encoding="utf-8")
                label_doc = json.loads(original_label)
            except (OSError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=409, detail="The video's label file could not be read safely.") from exc
            if not isinstance(label_doc, dict):
                raise HTTPException(status_code=409, detail="The video's label file has an invalid format.")

        videos_root.mkdir(parents=True, exist_ok=True)
        labels_root.mkdir(parents=True, exist_ok=True)
        with cache_lock:
            try:
                source.rename(destination)
                if old_label.exists():
                    old_label.rename(new_label)
                    label_doc["video"] = destination.name
                    new_label.write_text(json.dumps(label_doc), encoding="utf-8")
                if old_backup.exists():
                    old_backup.rename(new_backup)
            except OSError as exc:
                if new_backup.exists() and not old_backup.exists():
                    new_backup.rename(old_backup)
                if new_label.exists() and not old_label.exists():
                    new_label.rename(old_label)
                    if original_label is not None:
                        old_label.write_text(original_label, encoding="utf-8")
                if destination.exists() and not source.exists():
                    destination.rename(source)
                raise HTTPException(status_code=500, detail=f"Could not rename video: {exc}") from exc
            video_cache.pop(source.name, None)
        return {"name": destination.name}

    @app.post("/api/models/{model_id}/rename")
    def rename_model(model_id: str, request: RenameRequest) -> dict[str, str]:
        source, root_index = _model_path(model_id, resolved_model_roots)
        if root_index != 0:
            raise HTTPException(status_code=403, detail="Models in read-only libraries cannot be renamed.")
        task_info = _model_task(source, root_index)
        if task_info is None:
            raise HTTPException(status_code=400, detail="This model does not have a supported training target.")
        _task, suffix = task_info
        new_stem = _validate_segment(request.new_name)
        if new_stem.lower().endswith(suffix.lower()):
            new_stem = new_stem[: -len(suffix)]
        elif new_stem.lower().endswith(".pt"):
            new_stem = new_stem[:-3]
        if not new_stem:
            raise HTTPException(status_code=400, detail="Enter a valid model name.")
        destination = source.with_name(f"{new_stem}{suffix}")
        if destination == source:
            return {"name": source.name}
        if destination.exists() or destination.is_symlink():
            raise HTTPException(status_code=409, detail="A model with that name already exists.")
        try:
            source.rename(destination)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not rename model: {exc}") from exc
        return {"name": destination.name}

    @app.delete("/api/models/{model_id}")
    def delete_model(model_id: str) -> dict[str, str]:
        path, root_index = _model_path(model_id, resolved_model_roots)
        if root_index != 0:
            raise HTTPException(status_code=403, detail="Models in read-only libraries cannot be deleted.")
        if _model_task(path, root_index) is None:
            raise HTTPException(status_code=400, detail="This model does not have a supported training target.")
        try:
            path.unlink()
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not delete model: {exc}") from exc
        return {"name": path.name}

    app.mount("/", StaticFiles(directory=PORTAL_DIR / "static", html=True), name="static-root")
    return app
