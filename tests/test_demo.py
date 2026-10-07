import io
import json
from pathlib import Path

from Demo.flask_app import DemoConfig, create_app


class FakeIndex:
    ntotal = 1


class FakeModel:
    def embed_text(self, text, is_query=True):
        return {"text": text, "is_query": is_query}


class FakeEngine:
    def __init__(self):
        self.index = FakeIndex()
        self.model = FakeModel()

    @staticmethod
    def _result():
        return {
            "type": "video",
            "path": "Train/action_7",
            "action_id": "Train_action_7",
            "score": 0.875,
        }

    def search(self, query_embedding, top_k=5):
        assert query_embedding["is_query"]
        assert top_k == 5
        return [self._result()]

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
    assert client.get("/health").get_json()["ready"] is True


def test_text_and_video_search(tmp_path):
    client = _make_app(tmp_path).test_client()

    text_response = client.post("/api/search_text", json={"query": "sliding tackle"})
    assert text_response.status_code == 200
    result = text_response.get_json()["results"][0]
    assert result["labels"]["action"] == "Tackling"
    assert result["score_display"] == "0.875"
    assert result["video_url"] == "/videos/Train/action_7/clip_0.mp4"
    assert client.get(result["video_url"]).data == b"gallery-video"

    video_response = client.post(
        "/api/search_video",
        data={"video": (io.BytesIO(b"query-video"), "incident.mp4")},
        content_type="multipart/form-data",
    )
    assert video_response.status_code == 200
    body = video_response.get_json()
    assert body["results"][0]["action_id"] == "Train_action_7"
    assert client.get(body["query_video_url"]).data == b"query-video"


def test_rejects_unsupported_inputs(tmp_path):
    client = _make_app(tmp_path, model="internvideo2-1b").test_client()

    response = client.post("/api/search_text", json={"query": "tackle"})
    assert response.status_code == 400
    assert "not supported" in response.get_json()["error"]

    response = client.post(
        "/api/search_video",
        data={"video": (io.BytesIO(b"not-mp4"), "incident.avi")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
