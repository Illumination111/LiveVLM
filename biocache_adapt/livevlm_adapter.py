"""Strict timestamp-driven adapter around LiveVLM's official StreamCache path."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .common import Budget, JsonlWriter, StageTimer, state_report, write_manifest
from .schema import BenchmarkQuery


@dataclass(frozen=True)
class TimestampedFrame:
    timestamp: float
    frame: Any


def _coerce_frame(item: Any) -> TimestampedFrame:
    if isinstance(item, TimestampedFrame):
        return item
    if isinstance(item, Mapping):
        return TimestampedFrame(float(item["timestamp"]), item.get("frame", item.get("image")))
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return TimestampedFrame(float(item[0]), item[1])
    raise TypeError("frame must be TimestampedFrame, mapping, or (timestamp, frame)")


class LiveVLMStreamAdapter:
    """Incremental LiveVLM evaluation adapter.

    Frames are decoded by the caller and pushed with timestamps. ``available_until``
    is mandatory: every frame must be at or before that watermark, and a query can
    only be answered at the current watermark. This makes accidental future-frame
    pre-read an explicit protocol error.
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        image_processor: Any = None,
        *,
        dry_run: bool = False,
        cache_size: int = 12000,
        recent_frames: int = 2,
        scatter_ratio: float = 0.5,
        chunk_size: int = 16,
        retr_ratio: float = 0.4,
        clip_frame_num: int = 32,
        max_tokens: int = 128,
        budget: Any = "25%",
        run_dir: Optional[str] = None,
    ) -> None:
        if not dry_run and any(item is None for item in (model, tokenizer, image_processor)):
            raise ValueError("model, tokenizer and image_processor are required unless dry_run=True")
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.dry_run = bool(dry_run)
        self.args = SimpleNamespace(
            cache_size=int(cache_size),
            recent_frames=int(recent_frames),
            scatter_ratio=float(scatter_ratio),
            chunk_size=int(chunk_size),
            retr_ratio=float(retr_ratio),
            max_tokens=int(max_tokens),
        )
        self.clip_frame_num = int(clip_frame_num)
        self.budget = Budget.from_value(budget)
        self.run_dir = run_dir
        self.timer = StageTimer()
        self._writer = None  # type: Optional[JsonlWriter]
        self._opened = False
        self.reset()

    def open_stream(self, stream_id: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        if self._opened:
            raise RuntimeError("a stream is already open")
        self.reset()
        self._opened = True
        self.stream_id = str(stream_id)
        self.metadata = dict(metadata or {})
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            self._writer = JsonlWriter(os.path.join(self.run_dir, "results.jsonl"))
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            write_manifest(
                os.path.join(self.run_dir, "manifest.json"),
                repo_root,
                {
                    "adapter": "livevlm",
                    "dry_run": self.dry_run,
                    "budget": asdict(self.budget),
                    "cache": vars(self.args),
                    "clip_frame_num": self.clip_frame_num,
                },
                {"stream_id": self.stream_id, "metadata": self.metadata},
            )

    def reset(self) -> None:
        self.stream_id = ""
        self.metadata = {}
        self.stream_cache = None
        self.stream_time = 0.0
        self.last_frame_timestamp = float("-inf")
        self.frames_pushed = 0
        self.queries_answered = 0
        self.last_answer_metrics = {}
        self.timer = StageTimer()

    def _official_functions(self) -> Tuple[Any, Any, Any]:
        # Delayed import keeps --help/dry-run independent of torch/decord/models.
        from modules.model_builder import format_mcqa_prompt
        from modules.tools import frame_encoding, response

        return frame_encoding, response, format_mcqa_prompt

    def push_frames(self, frames: Iterable[Any], available_until: float) -> Dict[str, Any]:
        if not self._opened:
            raise RuntimeError("open_stream must be called first")
        watermark = float(available_until)
        if watermark + 1e-9 < self.stream_time:
            raise ValueError("stream watermark cannot move backwards")
        normalized = [_coerce_frame(item) for item in frames]
        for item in normalized:
            if item.timestamp > watermark + 1e-9:
                raise ValueError("future frame exceeds available_until watermark")
            if item.timestamp + 1e-9 < self.last_frame_timestamp:
                raise ValueError("frames must be pushed in nondecreasing timestamp order")
            self.last_frame_timestamp = item.timestamp
        with self.timer.measure("frame_prefill", frames=len(normalized), until=watermark):
            if not self.dry_run and normalized:
                frame_encoding, _, _ = self._official_functions()
                for start in range(0, len(normalized), self.clip_frame_num):
                    raw = [item.frame for item in normalized[start : start + self.clip_frame_num]]
                    pixels = self.image_processor.preprocess(raw, return_tensors="pt")["pixel_values"]
                    pixels = pixels.half().to(self.model.device)
                    self.stream_cache = frame_encoding(
                        [pixels], self.model, self.tokenizer, self.stream_cache, self.args
                    )
                    self.stream_cache.enable_biocache_observability = True
        self.frames_pushed += len(normalized)
        self.stream_time = watermark
        return {"frames": len(normalized), "available_until": watermark}

    def answer(self, query: Any) -> str:
        if not self._opened:
            raise RuntimeError("open_stream must be called first")
        if isinstance(query, BenchmarkQuery):
            timestamp, question, options, query_id = (
                query.timestamp, query.question, query.options, query.query_id
            )
        elif isinstance(query, Mapping):
            timestamp = float(query.get("timestamp", query.get("query_time")))
            question = str(query.get("question", query.get("query", "")))
            options = query.get("options", query.get("choices"))
            query_id = str(query.get("query_id", query.get("id", self.queries_answered)))
        else:
            raise TypeError("query must be BenchmarkQuery or mapping")
        if abs(float(timestamp) - self.stream_time) > 1e-9:
            raise ValueError(
                "query timestamp must equal current stream watermark; push only frames "
                "available at this query before answering"
            )
        with self.timer.measure("query", query_id=query_id, timestamp=float(timestamp)):
            if self.dry_run:
                output = "[dry-run]"
            else:
                _, response, format_mcqa_prompt = self._official_functions()
                prompt = format_mcqa_prompt(question, options) if options else question
                output = response(
                    None,
                    prompt,
                    question,
                    self.model,
                    self.tokenizer,
                    self.stream_cache,
                    self.args.max_tokens,
                    self.args,
                )
        self.queries_answered += 1
        cache_stats = getattr(self.stream_cache, "last_temporary_cache_stats", {}) if self.stream_cache else {}
        traces = cache_stats.get("retrieval_trace", [])
        available = sum(item.get("available_chunks", 0) for item in traces)
        selected = sum(item.get("selected_chunks", 0) for item in traces)
        self.last_answer_metrics = {
            "temporary_cache": cache_stats,
            "retrieval_selected_chunks": selected,
            "retrieval_available_chunks": available,
            "retrieval_hit_ratio": (selected / available) if available else 0.0,
        }
        record = {
            "stream_id": self.stream_id,
            "query_id": query_id,
            "timestamp": float(timestamp),
            "answer": output,
            "metrics": self.last_answer_metrics,
            "state_bytes": self.state_bytes(),
        }
        if self._writer:
            self._writer.write(record)
        return output

    query = answer

    def persistent_state(self) -> Mapping[str, Any]:
        return {"stream_cache": self.stream_cache}

    def static_state(self) -> Mapping[str, Any]:
        return {"model": self.model, "budget": self.budget, "config": vars(self.args)}

    def temporary_state(self) -> Mapping[str, Any]:
        cache = self.stream_cache
        return {
            "key_cache": getattr(cache, "tmp_key_cache", None),
            "value_cache": getattr(cache, "tmp_value_cache", None),
            "query_cache": getattr(cache, "tmp_query_cache", None),
            "last_cache_stats": self.last_answer_metrics.get("temporary_cache", {}),
        }

    def state_bytes(self) -> Dict[str, Any]:
        return state_report(self.persistent_state(), self.static_state(), self.temporary_state())

    def close_stream(self) -> Dict[str, Any]:
        if not self._opened:
            return {}
        summary = {
            "stream_id": self.stream_id,
            "frames_pushed": self.frames_pushed,
            "queries_answered": self.queries_answered,
            "timing": self.timer.summary(),
            "state_bytes": self.state_bytes(),
            "last_answer_metrics": self.last_answer_metrics,
        }
        if self.run_dir:
            import json

            with open(os.path.join(self.run_dir, "summary.json"), "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
        self._opened = False
        return summary
