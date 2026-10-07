"""Flask UI for searching a SoccerNet-FoulRet Train index.

Run from the repository root after preparing the data and building an index:

    python Demo/flask_app.py --model qwen3vl-2b

The model and retrieval-engine imports are intentionally lazy. This keeps the
module importable for lightweight web tests and lets the loading page render
while large model weights are initialized in a background thread.
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from typing import Callable, Optional
import uuid

from flask import Flask, abort, jsonify, render_template, request, send_from_directory, url_for


DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DISPLAY_NAMES = {
    "qwen3vl-2b": "Qwen3-VL-Embedding 2B",
    "qwen3vl-8b": "Qwen3-VL-Embedding 8B",
    "qwen3vl-2b-finetuned": "Fine-tuned Qwen3-VL-Embedding 2B",
    "vlm2vec-v2": "VLM2Vec-V2",
    "lamra": "LamRA-Ret",
    "internvideo2-1b": "InternVideo2-Stage2 1B",
    "internvideo2-6b": "InternVideo2-Stage2 6B",
}


@dataclass(frozen=True)
class DemoConfig:
    """Runtime configuration for the local demo."""

    model: str = "qwen3vl-2b"
    data_dir: Path = PROJECT_ROOT / "data" / "mvfouls_window"
    index_dir: Optional[Path] = None
    lora_path: Optional[Path] = None
    clip_mode: str = "multi_clip"
    fps: Optional[float] = None
    max_frames: Optional[int] = None
    nframes: Optional[int] = 16
    top_k: int = 5
    host: str = "127.0.0.1"
    port: int = 5000
    max_upload_mb: int = 512

    @property
    def resolved_data_dir(self) -> Path:
        return self.data_dir.expanduser().resolve()

    @property
    def resolved_index_dir(self) -> Path:
        base = self.index_dir or PROJECT_ROOT / "retrieval" / "indexes" / self.model
        return base.expanduser().resolve()

class DemoState:
    """Own the lazily initialized model, index, annotations, and uploads."""

    def __init__(
        self,
        config: DemoConfig,
        engine_factory: Optional[Callable[[], object]] = None,
    ) -> None:
        self.config = config
        self._engine_factory = engine_factory or self._build_engine
        self._engine = None
        self._engine_error: Optional[str] = None
        self._loading = False
        self._loading_stage = "Waiting to start initialization…"
        self._loading_step = 0
        self._loading_total_steps = 3
        self._loading_started_at: Optional[float] = None
        self._lock = threading.Lock()
        self._annotations = None
        self.upload_dir = Path(tempfile.mkdtemp(prefix="sn_foulret_uploads_"))
        atexit.register(shutil.rmtree, self.upload_dir, ignore_errors=True)

    @property
    def engine(self):
        return self._engine

    @property
    def loading(self) -> bool:
        return self._loading

    @property
    def error(self) -> Optional[str]:
        return self._engine_error

    @property
    def ready(self) -> bool:
        return self._engine is not None

    @property
    def loading_stage(self) -> str:
        return self._loading_stage

    @property
    def loading_step(self) -> int:
        return self._loading_step

    @property
    def loading_total_steps(self) -> int:
        return self._loading_total_steps

    @property
    def loading_elapsed_seconds(self) -> int:
        if self._loading_started_at is None:
            return 0
        return max(0, int(time.monotonic() - self._loading_started_at))

    def _set_loading_stage(self, step: int, message: str) -> None:
        self._loading_step = step
        self._loading_stage = message
        logging.info("Demo initialization: %s", message)

    def _build_engine(self):
        # Heavy dependencies stay out of module import and the Flask reloader.
        from retrieval.engine import RetrievalEngine
        from retrieval.models import get_model, model_requires_lora

        data_dir = self.config.resolved_data_dir
        index_dir = self.config.resolved_index_dir
        embedding_path = index_dir / "embeddings_train.json"
        faiss_path = index_dir / "faiss_index_train.bin"

        self._set_loading_stage(1, "Checking the prepared data and Train index…")
        missing = [str(path) for path in (embedding_path, faiss_path) if not path.is_file()]
        if missing:
            build = (
                f"python -m retrieval.build_full_index --model {self.config.model} "
                f"--data_dir {data_dir} --index_dir {index_dir} --nframes 16"
            )
            raise FileNotFoundError(
                "The demo index is missing: " + ", ".join(missing) + f". Build it with: {build}"
            )

        model_kwargs = {
            "video_root": str(data_dir),
            "clip_mode": self.config.clip_mode,
        }
        if self.config.fps is not None:
            model_kwargs["fps"] = self.config.fps
        if self.config.max_frames is not None:
            model_kwargs["max_frames"] = self.config.max_frames
        if self.config.nframes is not None:
            frame_key = "num_frames" if self.config.model.startswith("internvideo2") else "nframes"
            model_kwargs[frame_key] = self.config.nframes
        if self.config.lora_path is not None:
            model_kwargs["lora_path"] = str(self.config.lora_path.expanduser().resolve())
        elif model_requires_lora(self.config.model):
            raise ValueError(f"--lora_path is required for model {self.config.model}")

        model_name = MODEL_DISPLAY_NAMES.get(self.config.model, self.config.model)
        self._set_loading_stage(2, f"Loading {model_name} weights onto the compute device…")
        model = get_model(self.config.model, **model_kwargs)
        self._set_loading_stage(3, "Loading the FAISS Train index…")
        return RetrievalEngine(
            model=model,
            project_root=str(data_dir),
            embedding_path=str(embedding_path),
            index_path=str(faiss_path),
        )

    def get_engine(self):
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            self._loading = True
            self._engine_error = None
            self._loading_started_at = time.monotonic()
            self._set_loading_stage(0, "Starting demo initialization…")
            try:
                self._engine = self._engine_factory()
                self._set_loading_stage(self._loading_total_steps, "Ready")
                logging.info("Loaded demo model and index for %s", self.config.model)
                return self._engine
            except Exception as exc:
                self._engine_error = str(exc)
                self._loading_stage = "Initialization failed"
                logging.exception("Could not initialize the demo")
                raise
            finally:
                self._loading = False

    def start_loading(self) -> None:
        if self.ready or self.loading:
            return

        def load() -> None:
            try:
                self.get_engine()
            except Exception:
                # get_engine records and logs the exception for /health.
                pass

        threading.Thread(target=load, name="demo-model-loader", daemon=True).start()

    def annotations(self) -> dict:
        if self._annotations is not None:
            return self._annotations
        annotation_path = self.config.resolved_data_dir / "Train" / "annotations.json"
        try:
            with annotation_path.open(encoding="utf-8") as handle:
                document = json.load(handle)
            self._annotations = document.get("Actions", document)
        except (FileNotFoundError, json.JSONDecodeError, AttributeError) as exc:
            logging.warning("Could not load Train annotations from %s: %s", annotation_path, exc)
            self._annotations = {}
        return self._annotations

    def format_results(self, results: list[dict]) -> list[dict]:
        annotations = self.annotations()
        formatted = []
        data_dir = self.config.resolved_data_dir

        for raw in results:
            item = dict(raw)
            relative_action = Path(str(item.get("path", "")))
            action_id = relative_action.name.removeprefix("action_")
            labels = annotations.get(action_id, {}) if isinstance(annotations, dict) else {}
            item["labels"] = {
                "action": str(labels.get("Action class", "")).strip(),
                "offence": str(labels.get("Offence", "")).strip(),
                "severity": str(labels.get("Severity", "")).strip(),
                "contact": str(labels.get("Contact", "")).strip(),
            }
            score = float(item.get("score", 0.0))
            item["score"] = score
            item["score_display"] = f"{score:.3f}"
            item["video_url"] = ""

            action_dir = (data_dir / relative_action).resolve()
            try:
                action_dir.relative_to(data_dir)
            except ValueError:
                logging.warning("Ignoring result path outside the data directory: %s", relative_action)
            else:
                clips = sorted(action_dir.glob("clip_*.mp4"))
                if clips:
                    rel_clip = clips[0].relative_to(data_dir).as_posix()
                    item["video_url"] = url_for("serve_dataset_video", filename=rel_clip)
            formatted.append(item)
        return formatted


def _json_error(message: str, status: int = 500):
    return jsonify({"error": message}), status


def create_app(
    config: Optional[DemoConfig] = None,
    engine_factory: Optional[Callable[[], object]] = None,
) -> Flask:
    """Create the Flask application, optionally with a test engine factory."""

    config = config or DemoConfig()
    app = Flask(__name__, template_folder=str(DEMO_DIR / "templates"))
    app.config["MAX_CONTENT_LENGTH"] = config.max_upload_mb * 1024 * 1024
    state = DemoState(config, engine_factory=engine_factory)
    app.extensions["foulret_demo"] = state

    @app.get("/health")
    def health():
        if state.ready:
            status = "ready"
        elif state.error:
            status = "error"
        elif state.loading:
            status = "loading"
        else:
            status = "initializing"
        return jsonify(
            status=status,
            ready=state.ready,
            model=config.model,
            error=state.error,
            stage=state.loading_stage,
            step=state.loading_step,
            total_steps=state.loading_total_steps,
            elapsed_seconds=state.loading_elapsed_seconds,
        )

    @app.get("/")
    def index():
        if not state.ready:
            state.start_loading()
            return render_template(
                "loading.html",
                model_display=MODEL_DISPLAY_NAMES.get(config.model, config.model),
            )

        total_indexed = getattr(getattr(state.engine, "index", None), "ntotal", 0)
        return render_template(
            "index.html",
            model_display=MODEL_DISPLAY_NAMES.get(config.model, config.model),
            total_indexed=f"{total_indexed:,}",
        )

    @app.post("/api/search_video")
    def search_video():
        upload = request.files.get("video")
        if upload is None or not upload.filename:
            return _json_error("Choose an MP4 video to upload.", 400)
        if Path(upload.filename).suffix.lower() != ".mp4":
            return _json_error("Only MP4 uploads are supported.", 400)

        upload_id = uuid.uuid4().hex
        action_dir = state.upload_dir / upload_id
        action_dir.mkdir(mode=0o700)
        video_path = action_dir / "clip_0.mp4"
        upload.save(video_path)

        try:
            results = state.get_engine().search_by_video(str(action_dir), top_k=config.top_k)
            return jsonify(
                results=state.format_results(results),
                query_video_url=url_for("serve_upload", upload_id=upload_id),
            )
        except Exception as exc:
            logging.exception("Video search failed")
            return _json_error(str(exc))

    @app.get("/videos/<path:filename>")
    def serve_dataset_video(filename: str):
        if not filename.lower().endswith(".mp4"):
            abort(404)
        return send_from_directory(config.resolved_data_dir, filename, mimetype="video/mp4", conditional=True)

    @app.get("/uploads/<upload_id>")
    def serve_upload(upload_id: str):
        if len(upload_id) != 32 or any(char not in "0123456789abcdef" for char in upload_id):
            abort(404)
        return send_from_directory(
            state.upload_dir / upload_id,
            "clip_0.mp4",
            mimetype="video/mp4",
            conditional=True,
        )

    @app.errorhandler(413)
    def upload_too_large(_error):
        return _json_error(f"Upload exceeds the {config.max_upload_mb} MB limit.", 413)

    return app


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    from retrieval.models import MODEL_REGISTRY

    parser = argparse.ArgumentParser(description="Run the SoccerNet-FoulRet local web demo")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="qwen3vl-2b")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data" / "mvfouls_window")
    parser.add_argument("--index_dir", type=Path, default=None)
    parser.add_argument("--lora_path", type=Path, default=None)
    parser.add_argument("--clip_mode", choices=("single_clip", "multi_clip"), default="multi_clip")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--nframes", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--max_upload_mb", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if args.top_k < 1:
        raise SystemExit("--top_k must be at least 1")
    if args.nframes is not None and args.nframes < 1:
        raise SystemExit("--nframes must be at least 1")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = DemoConfig(**vars(args))
    app = create_app(config)
    app.run(host=config.host, port=config.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
