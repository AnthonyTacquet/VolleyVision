from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .datasets import discover_calibrations, discover_sources
from .pipeline import run_pipeline

APP_DIR = Path(__file__).resolve().parent
SOURCE_ROOTS = [Path(p) for p in os.getenv("SOURCE_ROOTS", "/sources").split(":") if p]
CALIBRATIONS_ROOT = Path(os.getenv("CALIBRATIONS_ROOT", "/calibrations"))
VIDEO_ROOT = Path(os.getenv("VIDEO_ROOT", "/videos"))
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT") or str(APP_DIR / "workspace"))

try:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
except PermissionError:
    WORKSPACE_ROOT = APP_DIR / "workspace"
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

JOBS_ROOT = WORKSPACE_ROOT / "jobs"
MODELS_ROOT = WORKSPACE_ROOT / "models"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)
MODELS_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Volley Ball Training Studio", version="1.1.0")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="yolo-trainer")
_lock = threading.RLock()
_jobs: dict[str, dict[str, Any]] = {}
_cancel_events: dict[str, threading.Event] = {}


class SourceSelection(BaseModel):
    dataset_id: str
    class_id: int | None = None
    calibration_id: str | None = None


class JobRequest(BaseModel):
    sources: list[SourceSelection] = Field(min_length=1)
    target: Literal["ball_tracking", "serve", "reception", "dig", "set", "spike", "block", "floor"] = "ball_tracking"
    mask_background: bool = False
    margin: float = Field(default=0.35, ge=0.0, le=1.5)
    ball_box_scale: float = Field(default=1.0, ge=0.25, le=4.0)
    split_mode: str = "group_by_source"
    seed: int = 42
    model: str = "yolo11n.pt"
    epochs: int = Field(default=100, ge=1, le=1000)
    imgsz: int = Field(default=640, ge=320, le=2048)
    batch: int = Field(default=2, ge=-1, le=256)
    workers: int = Field(default=0, ge=0, le=32)
    patience: int = Field(default=25, ge=0, le=300)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_file(job_id: str) -> Path:
    return JOBS_ROOT / job_id / "job.json"


def _persist(job: dict[str, Any]) -> None:
    job["updated_at"] = _now()
    path = _job_file(job["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=2), encoding="utf-8")


def _append_log(job_id: str, message: str) -> None:
    stamp = datetime.now().astimezone().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    with _lock:
        log_path = JOBS_ROOT / job_id / "job.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        job = _jobs.get(job_id)
        if job:
            job["last_message"] = message
            _persist(job)


def _set_status(job_id: str, status: str, stage: str | None = None) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["status"] = status
        if stage is not None:
            job["stage"] = stage
        if status in {"completed", "failed"}:
            job["finished_at"] = _now()
        _persist(job)


def _clear_cancel(job_id: str) -> None:
    with _lock:
        _cancel_events.pop(job_id, None)


def _run_job(job_id: str, config: dict[str, Any]) -> None:
    cancel_event = threading.Event()
    with _lock:
        _cancel_events[job_id] = cancel_event
    try:
        _set_status(job_id, "preparing", "preparing")
        _append_log(job_id, "Job started. This container is configured to reserve an NVIDIA GPU.")
        result = run_pipeline(
            config=config,
            job_dir=JOBS_ROOT / job_id,
            output_root=WORKSPACE_ROOT,
            log=lambda message: _append_log(job_id, message),
            set_status=lambda status, stage=None: _set_status(job_id, status, stage),
            cancel_event=cancel_event,
        )
        with _lock:
            job = _jobs[job_id]
            job["status"] = "completed"
            job["stage"] = "completed"
            job["result"] = result
            job["finished_at"] = _now()
            _persist(job)
        _append_log(job_id, "All stages completed successfully.")
    except Exception as exc:
        if cancel_event.is_set():
            _set_status(job_id, "cancelled", "cancelled")
            _append_log(job_id, "Training cancelled by user.")
        else:
            _append_log(job_id, f"ERROR: {exc}")
            _append_log(job_id, "See the full traceback below.")
            import traceback
            _append_log(job_id, traceback.format_exc())
            _set_status(job_id, "failed", "failed")
        with _lock:
            job = _jobs.get(job_id)
            if job:
                job["error"] = str(exc) if not cancel_event.is_set() else "Cancelled by user"
                _persist(job)
    finally:
        _clear_cancel(job_id)


def _load_existing_jobs() -> None:
    for path in JOBS_ROOT.glob("*/job.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            if job.get("status") in {"queued", "preparing", "training"}:
                job["status"] = "interrupted"
                job["stage"] = "interrupted"
                job["error"] = "Container restarted while this job was running. Start a new job to resume."
                _persist(job)
            _jobs[job["id"]] = job
        except Exception:
            continue


_load_existing_jobs()


def _current_discovery() -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
    datasets, warnings = discover_sources(SOURCE_ROOTS, [VIDEO_ROOT])
    calibrations = discover_calibrations(CALIBRATIONS_ROOT)
    calibration_by_stem = {c.get("video_stem"): c for c in calibrations}
    for dataset in datasets:
        video_path = dataset.get("video_path")
        if video_path:
            match = calibration_by_stem.get(Path(video_path).stem)
            if match:
                dataset["suggested_calibration_id"] = match["id"]
    return datasets, calibrations, warnings


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")


def _read_gpu_usage() -> dict[str, Any]:
    try:
        if not torch.cuda.is_available():
            return {"available": False, "gpus": []}
        query = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {"available": False, "gpus": []}
        gpus: list[dict[str, Any]] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            index, name, util, mem_used, mem_total, temp = parts[:6]
            try:
                gpus.append({
                    "index": int(index),
                    "name": name,
                    "utilization_gpu": float(util),
                    "mem_used_mb": float(mem_used),
                    "mem_total_mb": float(mem_total),
                    "temp_c": float(temp),
                    "mem_free_mb": max(0.0, float(mem_total) - float(mem_used)),
                })
            except ValueError:
                continue
        return {"available": True, "gpus": gpus}
    except Exception:
        return {"available": False, "gpus": []}


@app.get("/api/status")
def status() -> dict[str, Any]:
    try:
        cuda = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda else None
        cuda_version = torch.version.cuda
    except Exception as exc:
        cuda = False
        gpu_name = None
        cuda_version = None
        gpu_error = str(exc)
    else:
        gpu_error = None
    gpu_usage = _read_gpu_usage()
    return {
        "cuda_available": cuda,
        "gpu_name": gpu_name,
        "compute_device": gpu_name if cuda else "CPU",
        "cuda_version": cuda_version,
        "gpu_error": gpu_error,
        "gpu_usage": gpu_usage,
        "parallel_jobs": 4,
        "source_roots": [str(p) for p in SOURCE_ROOTS],
        "calibrations_root": str(CALIBRATIONS_ROOT),
        "video_root": str(VIDEO_ROOT),
        "workspace_root": str(WORKSPACE_ROOT),
    }


@app.get("/api/datasets")
def datasets() -> dict[str, Any]:
    found, calibrations, warnings = _current_discovery()
    public = []
    for item in found:
        public.append({k: item.get(k) for k in ("id", "name", "source_type", "yaml_path", "annotation_path", "video_path", "video_available", "video_reference", "root", "classes", "suggested_class", "suggested_calibration_id", "counts", "images", "labelled_images", "ball_visible", "ball_invisible", "invalid_points", "event_count", "event_types", "frame_size")})
    return {"datasets": public, "calibrations": calibrations, "warnings": warnings}


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    with _lock:
        result = sorted(_jobs.values(), key=lambda x: x.get("created_at", ""), reverse=True)
        return {"jobs": result[:30]}


@app.post("/api/jobs")
def create_job(request: JobRequest) -> dict[str, Any]:
    found, calibrations, _warnings = _current_discovery()
    by_id = {d["id"]: d for d in found}
    cal_by_id = {c["id"]: c for c in calibrations}
    selections: list[dict[str, Any]] = []
    for item in request.sources:
        dataset = by_id.get(item.dataset_id)
        if not dataset:
            raise HTTPException(status_code=400, detail=f"Dataset not found: {item.dataset_id}. Refresh the list.")
        source_type = dataset.get("source_type", "yolo")
        if request.target != "ball_tracking" and source_type != "volley_json":
            raise HTTPException(status_code=400, detail=f"Event training requires Volley Labeler JSON annotations: {dataset['name']}.")
        if request.target == "ball_tracking" and source_type == "yolo":
            if item.class_id is None or item.class_id not in {c["id"] for c in dataset["classes"]}:
                raise HTTPException(status_code=400, detail=f"Choose a valid ball class for {dataset['name']}.")
        elif source_type == "volley_json":
            if not dataset.get("video_path") or not Path(dataset["video_path"]).is_file():
                raise HTTPException(status_code=400, detail=f"Source video is missing for {dataset['name']}. Check VIDEO_HOST_PATH.")
        calibration_id = item.calibration_id or dataset.get("suggested_calibration_id")
        calibration_path = None
        if calibration_id:
            calibration = cal_by_id.get(calibration_id)
            if not calibration:
                raise HTTPException(status_code=400, detail=f"Court calibration not found: {calibration_id}.")
            calibration_path = calibration["path"]
        selections.append({"dataset_id": item.dataset_id, "source_type": source_type,
                           "class_id": item.class_id if item.class_id is not None else 0,
                           "calibration_path": calibration_path})

    if request.mask_background and any(not s["calibration_path"] for s in selections):
        raise HTTPException(status_code=400, detail="Background filtering requires a court.json selection for every selected dataset.")

    job_id = uuid.uuid4().hex[:10]
    config = request.model_dump()
    config["sources"] = selections
    config["source_roots"] = [str(p) for p in SOURCE_ROOTS]
    config["calibrations_root"] = str(CALIBRATIONS_ROOT)
    config["video_root"] = str(VIDEO_ROOT)
    job = {
        "id": job_id,
        "status": "queued",
        "stage": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        "settings": {k: v for k, v in config.items() if k not in {"source_roots", "calibrations_root", "video_root", "sources"}},
        "sources": [by_id[s["dataset_id"]]["name"] for s in selections],
        "last_message": "Queued for training",
    }
    with _lock:
        _jobs[job_id] = job
        _persist(job)
    _executor.submit(_run_job, job_id, config)
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.get("status") in {"completed", "failed", "cancelled", "interrupted"}:
            return {"ok": False, "status": job.get("status"), "message": "This job is already finished."}
        job["status"] = "canceling"
        job["stage"] = "canceling"
        job["updated_at"] = _now()
        _persist(job)
        event = _cancel_events.get(job_id)
        if event is not None:
            event.set()
        else:
            job["status"] = "cancelled"
            job["stage"] = "cancelled"
            job["error"] = "Cancelled by user"
            _persist(job)
    return {"ok": True, "status": "canceling"}


@app.get("/api/jobs/{job_id}/logs")
def get_job_logs(job_id: str) -> dict[str, str]:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    log_path = JOBS_ROOT / job_id / "job.log"
    if not log_path.exists():
        return {"logs": ""}
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"logs": "\n".join(lines[-500:])}


@app.get("/api/jobs/{job_id}/artifact/{artifact_name}")
def download_artifact(job_id: str, artifact_name: str) -> FileResponse:
    job = _jobs.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Completed job not found")
    result = job.get("result") or {}
    allowed = {
        "best.pt": result.get("model_path"),
        "track_ball.py": result.get("tracking_script"),
        "bundle.zip": result.get("bundle_path"),
    }
    path_value = allowed.get(artifact_name)
    if not path_value:
        raise HTTPException(status_code=404, detail="Artifact not found")
    path = Path(path_value).resolve()
    if not path.is_file() or WORKSPACE_ROOT.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail="Artifact file is missing")
    media_type = "application/octet-stream" if artifact_name.endswith(".pt") else "application/zip" if artifact_name.endswith(".zip") else "text/x-python"
    return FileResponse(path, filename=path.name, media_type=media_type)
