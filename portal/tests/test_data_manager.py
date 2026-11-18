import json
import threading

from fastapi.testclient import TestClient

from portal.data_manager import create_data_manager_app


def test_video_label_import_export_and_delete_are_type_aware(tmp_path):
    videos = tmp_path / "videos"
    labels = tmp_path / "labels"
    videos.mkdir()
    labels.mkdir()
    (videos / "match.mp4").write_bytes(b"video")
    label_path = labels / "match.mp4.json"
    original = {
        "video": "match.mp4",
        "rev": 1,
        "ball": {"12": {"x": 20, "y": 30, "v": 1, "s": 16}},
        "events": [{"id": 1, "frame": 30, "type": "serve"}],
    }
    label_path.write_text(json.dumps(original), encoding="utf-8")
    client = TestClient(create_data_manager_app(videos, labels, [], {}, threading.Lock()))

    listed = client.get("/api/labels", params={"type": "ball"})
    assert listed.status_code == 200
    assert listed.json()["labels"][0]["ball_labels"] == 1

    exported = client.get("/api/labels/match.mp4/download", params={"type": "ball"})
    assert exported.status_code == 200
    assert exported.json()["ball"] == original["ball"]
    assert exported.json()["events"] == []

    replacement = {
        "video": "match.mp4",
        "ball": {},
        "events": [{"id": 2, "frame": 60, "type": "attack"}],
    }
    imported = client.post(
        "/api/labels/import",
        data={"target": "events"},
        files={"file": ("event-labels.json", json.dumps(replacement), "application/json")},
    )
    assert imported.status_code == 200
    saved = json.loads(label_path.read_text(encoding="utf-8"))
    assert saved["ball"] == original["ball"]
    assert saved["events"] == replacement["events"]

    deleted_ball = client.delete("/api/labels/match.mp4", params={"type": "ball"})
    assert deleted_ball.status_code == 200
    saved = json.loads(label_path.read_text(encoding="utf-8"))
    assert saved["ball"] == {}
    assert saved["events"] == replacement["events"]

    deleted_all = client.delete("/api/labels/match.mp4")
    assert deleted_all.status_code == 200
    assert not label_path.exists()
