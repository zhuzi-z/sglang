# SGLang profile Replay Tool

Replay selected schedule batches from JSONL files and profile them with `nsys` or `torch`.

## Data

You can get the input data from `tools/benchmark`.

Required files:

- `schedule_batch_file`
- `request_tokens_file` (optional, but recommended)

## Usage

### NSYS

```bash
nsys profile -t cuda,nvtx -c cudaProfilerApi --capture-range-end="repeat" \
python main.py \
    --schedule-batch-file="/path/to/batch/file" \
    --request-tokens-file="/path/to/request/tokens/file" \
    --server-args='{
        "model_path": "Qwen/Qwen3-8B",
        "load_format": "dummy",
        "disable_overlap_schedule": false
    }' \
    --profiler=nsys
```

### Torch Profiler

```bash
python main.py \
    --schedule-batch-file="/path/to/batch/file" \
    --request-tokens-file="/path/to/request/tokens/file" \
    --server-args='{
        "model_path": "Qwen/Qwen3-8B",
        "load_format": "dummy",
        "disable_overlap_schedule": false
    }' \
    --profiler=torch
```

## Notes

- `--schedule-row-number-filter` selects which schedule rows to replay.
- The script forces `disable_overlap_schedule = True` during profiling.
