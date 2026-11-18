from __future__ import annotations

import base64
import csv
import json
import os
import subprocess
import threading
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from ultralytics import YOLO

from .compute import select_compute_device

APP_DIR = Path(__file__).resolve().parent
VIDEO_ROOT = Path(os.getenv("VIDEO_ROOT", "/videos")).resolve()
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", "/workspace")).resolve()
MODELS_ROOT = Path(os.getenv("MODELS_ROOT", str(WORKSPACE_ROOT / "models"))).resolve()
EXTRA_VIDEO_ROOTS = [Path(value).resolve() for value in os.getenv("EXTRA_VIDEO_ROOTS", "").split(":") if value]
EXTRA_MODEL_ROOTS = [Path(value).resolve() for value in os.getenv("EXTRA_MODEL_ROOTS", "").split(":") if value]
VIDEO_ROOTS = [VIDEO_ROOT, *EXTRA_VIDEO_ROOTS]
MODEL_ROOTS = [MODELS_ROOT, *EXTRA_MODEL_ROOTS]
JOBS_ROOT = WORKSPACE_ROOT / "analysis"
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".mpg", ".mpeg"}
TRACKERS = {"ocsort.yaml", "bytetrack.yaml"}
MODEL_TARGETS = ("ball_tracking", "serve", "reception", "dig", "set", "spike", "block", "floor")
JOBS_ROOT.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="VolleyVision Match Analyzer", version="1.0.0")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ball-analyzer")
_lock = threading.RLock()
_jobs: dict[str, dict[str, Any]] = {}


class AnalysisRequest(BaseModel):
    video_id: str
    models: dict[str, str] = Field(default_factory=dict)
    model_id: str | None = None
    tracker: str = "ocsort.yaml"
    confidence: float = Field(default=0.03, ge=0.001, le=1.0)
    imgsz: int = Field(default=960, ge=320, le=2048)
    trail_frames: int = Field(default=6, ge=1, le=30)


class TrailRecalibrationRequest(BaseModel):
    trail_frames: int = Field(ge=1, le=30)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_file(job_id: str) -> Path:
    return JOBS_ROOT / job_id / "job.json"


def _persist(job: dict[str, Any]) -> None:
    job["updated_at"] = _now()
    path = _job_file(job["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=2), encoding="utf-8")


def _log(job_id: str, message: str) -> None:
    stamp = datetime.now().astimezone().strftime("%H:%M:%S")
    with _lock:
        log_path = JOBS_ROOT / job_id / "job.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {message}\n")
        job = _jobs.get(job_id)
        if job:
            job["last_message"] = message
            _persist(job)


def _update(job_id: str, **values: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.update(values)
            _persist(job)


def _encode_video_id(root_index: int, relative_path: str) -> str:
    value = f"{root_index}:{relative_path}"
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _video_path(video_id: str) -> tuple[Path, str]:
    try:
        padding = "=" * (-len(video_id) % 4)
        value = base64.urlsafe_b64decode(video_id + padding).decode("utf-8")
        root_text, relative = value.split(":", 1)
        root_index = int(root_text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid video identifier") from exc
    pure = PurePosixPath(relative)
    if root_index < 0 or root_index >= len(VIDEO_ROOTS) or pure.is_absolute() or ".." in pure.parts:
        raise HTTPException(status_code=400, detail="Invalid video path")
    root = VIDEO_ROOTS[root_index]
    path = (root / Path(*pure.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Video path is outside the video directory") from exc
    if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Video not found")
    display_path = pure.as_posix() if root_index == 0 else f"library-{root_index}/{pure.as_posix()}"
    return path, display_path


def _model_path(model_id: str) -> Path:
    try:
        padding = "=" * (-len(model_id) % 4)
        value = base64.urlsafe_b64decode(model_id + padding).decode("utf-8")
        root_text, relative = value.split(":", 1)
        root_index = int(root_text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid model selection") from exc
    pure = PurePosixPath(relative)
    if root_index < 0 or root_index >= len(MODEL_ROOTS) or pure.is_absolute() or ".." in pure.parts or pure.suffix != ".pt":
        raise HTTPException(status_code=400, detail="Invalid model selection")
    root = MODEL_ROOTS[root_index]
    path = (root / Path(*pure.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Model path is outside the models directory") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Model not found. Refresh the model list.")
    return path


def _model_root_index(model_id: str) -> int:
    try:
        padding = "=" * (-len(model_id) % 4)
        root_text, _relative = base64.urlsafe_b64decode(model_id + padding).decode("utf-8").split(":", 1)
        return int(root_text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid model selection") from exc


def _model_target(path: Path, root_index: int) -> str | None:
    task = next((target for target in MODEL_TARGETS if path.name.endswith(f"_{target}_best.pt")), None)
    if task:
        return task
    if path.name.endswith("_ball_best.pt") or root_index > 0:
        return "ball_tracking"
    return None


def _load_jobs() -> None:
    for path in JOBS_ROOT.glob("*/job.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            if job.get("status") in {"queued", "analyzing"}:
                job["status"] = "interrupted"
                job["stage"] = "interrupted"
                job["last_message"] = "Container restarted while this analysis was running. Start a new analysis."
                _persist(job)
            if job.get("render_status") in {"queued", "recalibrating"}:
                job["render_status"] = "failed"
                job["render_error"] = "Container restarted during trail recalibration. Try again."
                _persist(job)
            _jobs[job["id"]] = job
        except Exception:
            continue


_load_jobs()


def _draw_ball_trails(
    frame: Any,
    trails: dict[int, deque[tuple[int, int, int]]],
    frame_number: int,
    trail_length: int,
) -> Any:
    trail_length = max(1, trail_length)
    radius = max(4, round(min(frame.shape[:2]) / 120))
    stale_ids = []
    for track_id, points in trails.items():
        while points and frame_number - points[0][0] >= trail_length:
            points.popleft()
        if not points:
            stale_ids.append(track_id)
            continue
        for observed_frame, x, y in points:
            intensity = 1 - (frame_number - observed_frame) / trail_length
            dot_radius = max(2, round(radius * intensity))
            center = (x, y)
            cv2.circle(frame, center, dot_radius + 2, (0, int(110 + 100 * intensity), 255),
                       -1, cv2.LINE_AA)
            cv2.circle(frame, center, dot_radius, (0, 255, 255), -1, cv2.LINE_AA)
    for track_id in stale_ids:
        del trails[track_id]
    return frame


def _run_analysis(job_id: str, video: Path, model_paths: dict[str, Path], settings: dict[str, Any]) -> None:
    capture = None
    encoder = None
    csv_handle = None
    events_handle = None
    try:
        device = select_compute_device(torch.cuda.is_available())
        device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
        _log(job_id, f"Running inference on {device_name}.")
        _update(job_id, status="analyzing", stage="loading", progress=0)
        models = {}
        for target, model_path in model_paths.items():
            _log(job_id, f"Loading {model_path.name} for {target.replace('_', ' ')} inference.")
            models[target] = YOLO(str(model_path))
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video.name}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise RuntimeError("Could not read video dimensions")
        if not fps or fps <= 0:
            fps = 25.0

        job_dir = JOBS_ROOT / job_id
        output_video = job_dir / "tracked.mp4"
        output_csv = job_dir / "tracks.csv"
        output_events = job_dir / "events.csv"
        encoder = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-video_size", f"{width}x{height}", "-framerate", str(fps),
             "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_video)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        csv_handle = output_csv.open("w", newline="", encoding="utf-8")
        rows = csv.writer(csv_handle)
        rows.writerow(["frame", "time_seconds", "track_id", "x1", "y1", "x2", "y2", "confidence"])
        events_handle = output_events.open("w", newline="", encoding="utf-8")
        event_rows = csv.writer(events_handle)
        event_rows.writerow(["frame", "time_seconds", "event", "x1", "y1", "x2", "y2", "confidence"])
        _update(job_id, stage="inference", total_frames=total_frames, width=width, height=height,
                fps=round(fps, 3), started_at=_now())
        _log(job_id, f"Analyzing {total_frames:,} frames at {width}x{height}, {fps:.3f} fps with {len(models)} selected model(s).")

        frame_number = 0
        trail_length = settings["trail_frames"]
        trails: dict[int, deque[tuple[int, int, int]]] = {}
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            inference_frame = frame.copy()
            time_seconds = frame_number / fps
            ball_model = models.get("ball_tracking")
            if ball_model:
                result = ball_model.track(
                    inference_frame.copy(), persist=True, tracker=settings["tracker"], classes=[0],
                    conf=settings["confidence"], imgsz=settings["imgsz"], device=device, verbose=False,
                )[0]
                boxes = result.boxes
                if boxes is not None and len(boxes):
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(xyxy)
                    for box, track_id, confidence in zip(xyxy, ids, confs):
                        center_x = int((box[0] + box[2]) / 2)
                        center_y = int((box[1] + box[3]) / 2)
                        trails.setdefault(int(track_id), deque(maxlen=trail_length)).append(
                            (frame_number, center_x, center_y)
                        )
                        rows.writerow([frame_number, round(time_seconds, 4), int(track_id),
                                       *[round(float(value), 2) for value in box], round(float(confidence), 5)])
            _draw_ball_trails(frame, trails, frame_number, trail_length)
            for target, event_model in models.items():
                if target == "ball_tracking":
                    continue
                result = event_model.predict(
                    inference_frame.copy(), classes=[0], conf=settings["confidence"], imgsz=settings["imgsz"],
                    device=device, verbose=False,
                )[0]
                boxes = result.boxes
                if boxes is None or not len(boxes):
                    continue
                for box, confidence in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = (int(value) for value in box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 220, 150), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"{target} {float(confidence):.2f}", (x1, max(20, y1 - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 220, 150), 2, cv2.LINE_AA)
                    event_rows.writerow([frame_number, round(time_seconds, 4), target,
                                         round(float(box[0]), 2), round(float(box[1]), 2),
                                         round(float(box[2]), 2), round(float(box[3]), 2),
                                         round(float(confidence), 5)])
            encoder.stdin.write(frame.tobytes())
            frame_number += 1
            if frame_number % 25 == 0 or frame_number == total_frames:
                progress = min(99, int(frame_number * 100 / total_frames)) if total_frames else 0
                _update(job_id, frames_done=frame_number, progress=progress)

        if frame_number == 0:
            raise RuntimeError("No readable frames were found in this video")
        capture.release()
        capture = None
        csv_handle.close()
        csv_handle = None
        events_handle.close()
        events_handle = None
        encoder.stdin.close()
        encoder.stdin = None
        return_code = encoder.wait()
        if return_code != 0:
            raise RuntimeError("Could not encode the annotated MP4. Check available disk space and ffmpeg.")
        encoder = None
        with _lock:
            job = _jobs[job_id]
            job.update(status="completed", stage="completed", progress=100, frames_done=frame_number,
                       result={"video": str(output_video), "csv": str(output_csv), "events": str(output_events)},
                       trail_frames=trail_length, video_revision=1, render_status="ready", finished_at=_now())
            _persist(job)
        _log(job_id, f"Completed: {frame_number:,} frames. Annotated video and track CSV are ready.")
    except Exception as exc:
        _log(job_id, f"ERROR: {exc}")
        _update(job_id, status="failed", stage="failed", error=str(exc), finished_at=_now())
    finally:
        if capture is not None:
            capture.release()
        if csv_handle is not None:
            csv_handle.close()
        if events_handle is not None:
            events_handle.close()
        if encoder is not None:
            try:
                if encoder.stdin:
                    encoder.stdin.close()
            except OSError:
                pass
            encoder.wait()


def _source_video_path(job: dict[str, Any]) -> Path:
    source_value = job.get("source_video")
    if source_value:
        candidate = Path(source_value).resolve()
        for root in VIDEO_ROOTS:
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            if candidate.is_file():
                return candidate
        raise HTTPException(status_code=404, detail="Original source video is unavailable")

    display_path = str(job.get("video", ""))
    if display_path.startswith("library-"):
        root_name, separator, relative_path = display_path.partition("/")
        try:
            root_index = int(root_name.removeprefix("library-"))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Original source video is unavailable") from exc
        if not separator:
            raise HTTPException(status_code=404, detail="Original source video is unavailable")
    else:
        root_index, relative_path = 0, display_path
    pure = PurePosixPath(relative_path)
    if root_index < 0 or root_index >= len(VIDEO_ROOTS) or pure.is_absolute() or ".." in pure.parts:
        raise HTTPException(status_code=404, detail="Original source video is unavailable")
    root = VIDEO_ROOTS[root_index]
    candidate = (root / Path(*pure.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Source video is outside the video library") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Original source video is unavailable")
    return candidate


def _job_tracks_path(job_id: str, job: dict[str, Any]) -> Path:
    csv_value = (job.get("result") or {}).get("csv")
    if not csv_value:
        raise HTTPException(status_code=404, detail="Saved track data is unavailable")
    path = Path(csv_value).resolve()
    try:
        path.relative_to((JOBS_ROOT / job_id).resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Invalid track data path") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Saved track data is unavailable")
    return path


def _recalibrate_job(job_id: str, trail_frames: int) -> None:
    capture = None
    encoder = None
    temp_video = JOBS_ROOT / job_id / "tracked.recalibrating.mp4"
    try:
        with _lock:
            job = dict(_jobs[job_id])
        _update(job_id, render_status="recalibrating")
        source_video = _source_video_path(job)
        tracks_path = _job_tracks_path(job_id, job)
        detections: dict[int, list[tuple[int, int, int]]] = {}
        with tracks_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                frame_index = int(row["frame"])
                track_id = int(float(row["track_id"]))
                center_x = round((float(row["x1"]) + float(row["x2"])) / 2)
                center_y = round((float(row["y1"]) + float(row["y2"])) / 2)
                detections.setdefault(frame_index, []).append((track_id, center_x, center_y))
        event_detections: dict[int, list[tuple[str, float, float, float, float, float]]] = {}
        events_value = (job.get("result") or {}).get("events")
        if events_value:
            events_path = Path(events_value).resolve()
            try:
                events_path.relative_to((JOBS_ROOT / job_id).resolve())
            except ValueError as exc:
                raise HTTPException(status_code=403, detail="Invalid saved event data path") from exc
            if events_path.is_file():
                with events_path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        frame_index = int(row["frame"])
                        event_detections.setdefault(frame_index, []).append((
                            row["event"], float(row["x1"]), float(row["y1"]),
                            float(row["x2"]), float(row["y2"]), float(row["confidence"]),
                        ))

        capture = cv2.VideoCapture(str(source_video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open source video: {source_video.name}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise RuntimeError("Could not read source video dimensions")

        temp_video.unlink(missing_ok=True)
        encoder = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-video_size", f"{width}x{height}", "-framerate", str(fps),
             "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp_video)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        trails: dict[int, deque[tuple[int, int, int]]] = {}
        frame_number = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for track_id, center_x, center_y in detections.get(frame_number, []):
                trails.setdefault(track_id, deque(maxlen=trail_frames)).append(
                    (frame_number, center_x, center_y)
                )
            _draw_ball_trails(frame, trails, frame_number, trail_frames)
            for event, x1, y1, x2, y2, confidence in event_detections.get(frame_number, []):
                top_left = (int(x1), int(y1))
                bottom_right = (int(x2), int(y2))
                cv2.rectangle(frame, top_left, bottom_right, (70, 220, 150), 2, cv2.LINE_AA)
                cv2.putText(frame, f"{event} {confidence:.2f}", (top_left[0], max(20, top_left[1] - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 220, 150), 2, cv2.LINE_AA)
            encoder.stdin.write(frame.tobytes())
            frame_number += 1
            if frame_number % 25 == 0 or frame_number == total_frames:
                progress = min(99, int(frame_number * 100 / total_frames)) if total_frames else 0
                _update(job_id, render_frames_done=frame_number, render_progress=progress)

        if frame_number == 0:
            raise RuntimeError("No readable frames were found in the source video")
        capture.release()
        capture = None
        encoder.stdin.close()
        encoder.stdin = None
        return_code = encoder.wait()
        encoder = None
        if return_code != 0:
            raise RuntimeError("Could not encode the recalibrated MP4. Check available disk space and ffmpeg.")

        output_video = Path((job.get("result") or {})["video"])
        os.replace(temp_video, output_video)
        with _lock:
            current_job = _jobs[job_id]
            current_job.update(trail_frames=trail_frames, video_revision=int(current_job.get("video_revision", 0)) + 1,
                               render_status="ready", render_progress=100, render_frames_done=frame_number,
                               render_error=None, requested_trail_frames=None)
            _persist(current_job)
        _log(job_id, f"Trail recalibrated to {trail_frames} frames without rerunning detection.")
    except Exception as exc:
        _update(job_id, render_status="failed", render_error=str(exc))
        _log(job_id, f"Trail recalibration failed: {exc}")
    finally:
        if capture is not None:
            capture.release()
        if encoder is not None:
            try:
                if encoder.stdin:
                    encoder.stdin.close()
            except OSError:
                pass
            encoder.wait()
        temp_video.unlink(missing_ok=True)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status() -> dict[str, Any]:
    try:
        cuda = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda else None
        cuda_version = torch.version.cuda
    except Exception as exc:
        cuda, gpu_name, cuda_version = False, None, None
        gpu_error = str(exc)
    else:
        gpu_error = None
        return {"cuda_available": cuda, "gpu_name": gpu_name,
            "compute_device": gpu_name if cuda else "CPU", "cuda_version": cuda_version,
            "gpu_error": gpu_error, "video_roots": [str(root) for root in VIDEO_ROOTS],
            "models_roots": [str(root) for root in MODEL_ROOTS]}


@app.get("/api/videos")
def videos() -> dict[str, Any]:
    found = []
    for root_index, root in enumerate(VIDEO_ROOTS):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            try:
                resolved = path.resolve()
                resolved.relative_to(root)
                if not resolved.is_file() or resolved.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                relative = resolved.relative_to(root).as_posix()
                stat = resolved.stat()
            except (OSError, ValueError):
                continue
            display_path = relative if root_index == 0 else f"library-{root_index}/{relative}"
            found.append({"id": _encode_video_id(root_index, relative), "name": resolved.name,
                          "relative_path": display_path, "size_bytes": stat.st_size})
    found.sort(key=lambda item: item["relative_path"].lower())
    return {"videos": found, "video_root": str(VIDEO_ROOT)}


@app.get("/api/models")
def models() -> dict[str, Any]:
    found = []
    for root_index, root in enumerate(MODEL_ROOTS):
        if not root.is_dir():
            continue
        candidates = root.glob("*_best.pt") if root_index == 0 else root.rglob("*.pt")
        for path in candidates:
            task = _model_target(path, root_index)
            if task is None:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            relative = path.relative_to(root).as_posix()
            model_id = base64.urlsafe_b64encode(f"{root_index}:{relative}".encode("utf-8")).decode("ascii").rstrip("=")
            found.append({"id": model_id, "name": path.name, "task": task,
                          "source": "training workspace" if root_index == 0 else f"model library {root_index}",
                          "size_bytes": stat.st_size, "modified_at": stat.st_mtime})
    found.sort(key=lambda item: item["modified_at"], reverse=True)
    return {"models": found, "models_root": str(MODELS_ROOT)}


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    with _lock:
        result = sorted(_jobs.values(), key=lambda item: item.get("created_at", ""), reverse=True)
        return {"jobs": result[:30]}


@app.post("/api/jobs")
def create_job(request: AnalysisRequest) -> dict[str, Any]:
    if request.tracker not in TRACKERS:
        raise HTTPException(status_code=400, detail="Unsupported tracker selection")
    video, relative_path = _video_path(request.video_id)
    selected_ids = dict(request.models)
    if request.model_id and "ball_tracking" not in selected_ids:
        selected_ids["ball_tracking"] = request.model_id
    if not selected_ids:
        raise HTTPException(status_code=400, detail="Select at least one trained model.")
    if set(selected_ids) - set(MODEL_TARGETS):
        raise HTTPException(status_code=400, detail="Unsupported model target selection.")
    model_paths = {target: _model_path(model_id) for target, model_id in selected_ids.items()}
    for target, model_id in selected_ids.items():
        path = model_paths[target]
        if _model_target(path, _model_root_index(model_id)) != target:
            raise HTTPException(status_code=400, detail=f"Selected model is not a {target.replace('_', ' ')} model.")
    with _lock:
        if any(job.get("status") in {"queued", "analyzing"} for job in _jobs.values()):
            raise HTTPException(status_code=409, detail="An analysis is already running. Wait for it to finish first.")
        if any(job.get("render_status") in {"queued", "recalibrating"} for job in _jobs.values()):
            raise HTTPException(status_code=409, detail="A trail recalibration is already running. Wait for it to finish first.")
        job_id = uuid.uuid4().hex[:10]
        job_dir = JOBS_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        selected_names = {target: path.name for target, path in model_paths.items()}
        job = {"id": job_id, "status": "queued", "stage": "queued", "created_at": _now(),
               "video": relative_path, "source_video": str(video), "model": ", ".join(selected_names.values()),
               "models": selected_names,
               "settings": request.model_dump(exclude={"video_id", "models", "model_id"}),
               "progress": 0, "frames_done": 0, "last_message": "Queued for analysis"}
        _jobs[job_id] = job
        _persist(job)
    _executor.submit(_run_analysis, job_id, video, model_paths, job["settings"])
    return job


@app.post("/api/jobs/{job_id}/recalibrate")
def recalibrate_job(job_id: str, request: TrailRecalibrationRequest) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job or job.get("status") != "completed":
            raise HTTPException(status_code=404, detail="Completed analysis not found")
        if job.get("render_status") in {"queued", "recalibrating"}:
            raise HTTPException(status_code=409, detail="Trail recalibration is already running")
        if any(
            other.get("status") in {"queued", "analyzing"}
            or other.get("render_status") in {"queued", "recalibrating"}
            for other in _jobs.values()
        ):
            raise HTTPException(status_code=409, detail="Another analysis or recalibration is already running")
        _source_video_path(job)
        _job_tracks_path(job_id, job)
        job.update(render_status="queued", render_progress=0, render_frames_done=0,
                   requested_trail_frames=request.trail_frames, render_error=None)
        _persist(job)
        response = dict(job)
    _executor.submit(_recalibrate_job, job_id, request.trail_frames)
    return response


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Analysis job not found")
        return job


@app.get("/api/jobs/{job_id}/logs")
def get_job_logs(job_id: str) -> dict[str, str]:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    log_path = JOBS_ROOT / job_id / "job.log"
    if not log_path.exists():
        return {"logs": ""}
    return {"logs": "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-300:])}


@app.get("/api/jobs/{job_id}/artifact/{artifact_name}")
def download_artifact(job_id: str, artifact_name: str) -> FileResponse:
    with _lock:
        job = _jobs.get(job_id)
        if not job or job.get("status") != "completed":
            raise HTTPException(status_code=404, detail="Completed analysis not found")
        result = job.get("result") or {}
        allowed = {"tracked.mp4": result.get("video"), "tracks.csv": result.get("csv"),
               "events.csv": result.get("events")}
    path_value = allowed.get(artifact_name)
    if not path_value:
        raise HTTPException(status_code=404, detail="Analysis artifact not found")
    path = Path(path_value).resolve()
    try:
        path.relative_to((JOBS_ROOT / job_id).resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Invalid artifact path") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Analysis artifact is missing")
    media_type = "video/mp4" if artifact_name.endswith(".mp4") else "text/csv"
    return FileResponse(path, filename=f"{Path(job['video']).stem}_{artifact_name}", media_type=media_type)