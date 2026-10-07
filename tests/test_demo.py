import io
import json
from pathlib import Path

from Demo.flask_app import DemoConfig, create_app


class FakeIndex:
    ntotal = 1


class FakeEngine:
    def __init__(self):
        self.index = FakeIndex()

    @staticmethod
    def _result():
        return {
            "type": "video",
            "path": "Train/action_7",
            "action_id": "Train_action_7",
            "score": 0.875,
        }

    def search_by_video(self, action_dir, top_k=5):
        assert (Path(action_dir) / "clip_0.mp4").read_bytes() == b"query-video"
        assert top_k == 5
        return [self._result()]


def _make_app(tmp_path, model="qwen3vl-2b"):
    data_dir = tmp_path / "mvfouls_window"
    action_dir = data_dir / "Train" / "action_7"
    action_dir.mkdir(parents=True)
    (action_dir / "clip_0.mp4").write_bytes(b"gallery-video")
    (data_dir / "Train" / "annotations.json").write_text(
        json.dumps(
            {
                "Actions": {
                    "7": {
                        "Action class": "Tackling",
                        "Offence": "Offence",
                        "Severity": "3.0",
                        "Contact": "With contact",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    app = create_app(
        DemoConfig(model=model, data_dir=data_dir),
        engine_factory=FakeEngine,
    )
    app.config["TESTING"] = True
    app.extensions["foulret_demo"].get_engine()
    return app


def test_landing_and_health(tmp_path):
    client = _make_app(tmp_path).test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert b"SoccerNet-FoulRet" in page.data
    assert b"Qwen3-VL-Embedding 2B" in page.data
    assert b"Search with a description" not in page.data
    assert client.post("/api/search_text", json={"query": "tackle"}).status_code == 404
    health = client.get("/health").get_json()
    assert health["ready"] is True
    assert health["stage"] == "Ready"
    assert health["step"] == health["total_steps"] == 3
    assert isinstance(health["elapsed_seconds"], int)


def test_video_search(tmp_path):
    client = _make_app(tmp_path).test_client()

    video_response = client.post(
        "/api/search_video",
        data={"video": (io.BytesIO(b"query-video"), "incident.mp4")},
        content_type="multipart/form-data",
    )
    assert video_response.status_code == 200
    body = video_response.get_json()
    assert body["results"][0]["action_id"] == "Train_action_7"
    assert body["results"][0]["labels"]["action"] == "Tackling"
    assert body["results"][0]["score_display"] == "0.875"
    assert body["results"][0]["video_url"] == "/videos/Train/action_7/clip_0.mp4"
    assert client.get(body["results"][0]["video_url"]).data == b"gallery-video"
    assert client.get(body["query_video_url"]).data == b"query-video"


def test_rejects_unsupported_inputs(tmp_path):
    client = _make_app(tmp_path, model="internvideo2-1b").test_client()

    response = client.post(
        "/api/search_video",
        data={"video": (io.BytesIO(b"not-mp4"), "incident.avi")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
