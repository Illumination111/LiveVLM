# LiveVLM BioCache adapter

This package adds a strict streaming evaluation surface without replacing the
official `benchmarks/mlvu/run_streaming_qa.py` runner.

## Contract

- `open_stream(stream_id, metadata)` / `reset()`
- `push_frames(timestamped_frames, available_until)`: frames are decoded by the
  caller. Timestamps must be monotonic and no frame may exceed the watermark.
- `answer(query)` (alias `query`): the query timestamp must equal the current
  watermark, preventing first-question full-video pre-encoding.
- `close_stream()`
- `persistent_state()`, `static_state()`, `temporary_state()`, and
  `state_bytes()` expose recursively measured state.

The adapter batches only externally supplied frames into the official
`frame_encoding` function. It uses the official `response` function for
answers. Additive `StreamCache` instrumentation reports temporary key/value/query
bytes and per-layer retrieval selections before the official rollback clears
them.

Budget tiers are `12.5%`, `25%`, and `50%`. Each run directory receives
`manifest.json`, `results.jsonl`, and `summary.json`, including phased timing and
actual cache-byte observations.

## Benchmark schema

`schema.load_benchmark()` accepts JSON or JSONL and normalizes common aliases:
`video_path|video|path`, `queries|conversations|questions`, and
`timestamp|query_time|time`. Every query must have a timestamp unless it is the
only query and the sample has a duration.

## Safe validation

No model, video, data, or weights are touched:

```bash
python -m biocache_adapt.run_dry --run-dir /tmp/livevlm-biocache-dry
python -m biocache_adapt.run_dry --help
```

Real execution requires the caller to load LiveVLM and decode frames. Construct
`LiveVLMStreamAdapter(model=..., tokenizer=..., image_processor=...)`, push only
frames available at each query timestamp, then call `answer`.
