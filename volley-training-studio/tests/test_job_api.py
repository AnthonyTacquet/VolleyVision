from fastapi.testclient import TestClient

from app import main
from app.main import app, _cancel_events, _jobs

client = TestClient(app)


def test_status_reports_gpu_usage_snapshot():
    response = client.get("/api/status")
    assert response.status_code == 200
    payload = response.json()
    assert "gpu_usage" in payload
    assert isinstance(payload["gpu_usage"], dict)
    assert "available" in payload["gpu_usage"]


def test_cancel_route_marks_running_job():
    job_id = "sample-job-123"
    _jobs[job_id] = {
        "id": job_id,
        "status": "training",
        "stage": "training",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
        "last_message": "Training",
    }
    _cancel_events[job_id] = __import__("threading").Event()

    response = client.post(f"/api/jobs/{job_id}/cancel")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert _jobs[job_id]["status"] == "canceling"
    assert _cancel_events[job_id].is_set() is True


def test_create_job_accepts_cpu_only_runtime(monkeypatch):
    dataset = {
        "id": "sample-dataset",
        "name": "Sample dataset",
        "source_type": "yolo",
        "classes": [{"id": 0}],
    }
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(main, "_current_discovery", lambda: ([dataset], [], []))
    monkeypatch.setattr(main, "_persist", lambda _job: None)
    monkeypatch.setattr(main._executor, "submit", lambda *_args: None)

    response = client.post(
        "/api/jobs",
        json={"sources": [{"dataset_id": "sample-dataset", "class_id": 0}]},
    )

    assert response.status_code == 200
    job = response.json()
    assert job["last_message"] == "Queued for training"
    main._jobs.pop(job["id"], None)
