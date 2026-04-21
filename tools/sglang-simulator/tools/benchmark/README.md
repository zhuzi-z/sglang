# Collect GPU Execution Time of `ScheduleBatch` in SGLang

This tool is mainly used to collect the GPU execution time of `ScheduleBatch` in SGLang.

## 1. Start the service

> Note: `--disable-overlap-schedule` is **required**.

```sh
SGL_HOOK_REQ_INFO_DIR=`pwd`/data \
python3 ./sgl_launch_server.py \
    --model-path="Qwen/Qwen3-30B-A3B" \
    --disable-overlap-schedule \
    --tp 2 --ep 2 \
    --load-format=dummy
```

### Notes

- `SGL_HOOK_REQ_INFO_DIR` specifies the directory where exported profiling data will be saved.
- `--disable-overlap-schedule` must be enabled, otherwise the collected timing data may not match the actual `ScheduleBatch` execution boundaries.

## 2. Run the benchmark

```sh
python3 -m sglang.bench_serving --backend sglang \
    --dataset-name sharegpt --num-prompts 2000 \
    --request-rate 100 \
    --host 127.0.0.1
```

## 3. Export profiling data

The service intercepts the `stop_profile` endpoint.
Once this endpoint is called, the server will export the collected profiling data.

The exported data will be saved in the directory specified by `SGL_HOOK_REQ_INFO_DIR`.

> Important: no profiling data will be exported unless this endpoint is triggered.

```sh
curl http://localhost:30000/stop_profile
```
