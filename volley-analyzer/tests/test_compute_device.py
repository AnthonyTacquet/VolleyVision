import os
import tempfile

import pytest
from fastapi.testclient import TestClient

_test_workspace = tempfile.TemporaryDirectory(prefix="volley-analyzer-tests-")
os.environ.setdefault("WORKSPACE_ROOT", _test_workspace.name)

from app import main
from app.compute import select_compute_device

client = TestClient(main.app)


@pytest.mark.parametrize(
    ("cuda_available", "expected"),
    [(True, 0), (False, "cpu")],
)
def test_select_compute_device(cuda_available, expected):
    assert select_compute_device(cuda_available) == expected


def test_create_job_accepts_cpu_only_runtime(monkeypatch, tmp_path):
    video = tmp_path / "sample.mp4"
    model = tmp_path / "model_ball_tracking_best.pt"
    video.touch()
    model.touch()

    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(main, "_video_path", lambda _video_id: (video, "sample.mp4"))
    monkeypatch.setattr(main, "_model_path", lambda _model_id: model)
    monkeypatch.setattr(main, "_model_root_index", lambda _model_id: 0)
    monkeypatch.setattr(main, "_model_target", lambda _path, _root_index: "ball_tracking")
    monkeypatch.setattr(main, "_persist", lambda _job: None)
    monkeypatch.setattr(main._executor, "submit", lambda *_args: None)

    response = client.post(
        "/api/jobs",
        json={"video_id": "sample-video", "models": {"ball_tracking": "sample-model"}},
    )

    assert response.status_code == 200
    job = response.json()
    assert job["last_message"] == "Queued for analysis"
    main._jobs.pop(job["id"], None)
