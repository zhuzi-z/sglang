import os

from sglang_simulator.simulation.sglang.worker import SGLangWorker
from sglang_simulator.dataset import DatasetArgs, get_dataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig
from transformers import AutoTokenizer

os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = (
    os.path.dirname(__file__) + "/assets/config_sglang.json"
)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from sglang_simulator.simulation.benchmark.load_balance import RoundRobinPolicy
from sglang_simulator.simulation.benchmark.pd_dissag import PDDisaggBenchmarkRunner


def _create_pd_workers(model_path, num_prefill=1, num_decode=1):
    from sglang.srt.server_args import ServerArgs

    prefill_workers = []
    for idx in range(num_prefill):
        prefill_workers.append(
            SGLangWorker(
                server_args=ServerArgs(
                    model_path=model_path,
                    load_format="dummy",
                    device="cpu",
                    max_total_tokens=100000,
                    disaggregation_mode="prefill",
                    disaggregation_transfer_backend="mooncake",
                ),
                name=f"prefill_{idx}",
            )
        )

    decode_workers = []
    for idx in range(num_decode):
        decode_workers.append(
            SGLangWorker(
                server_args=ServerArgs(
                    model_path=model_path,
                    load_format="dummy",
                    device="cpu",
                    max_total_tokens=100000,
                    disaggregation_mode="decode",
                    disaggregation_transfer_backend="mooncake",
                ),
                name=f"decode_{idx}",
            )
        )

    return prefill_workers, decode_workers


def _create_dataset(model_path, num_prompts=10):
    dataset_args = DatasetArgs(
        "random_ids",
        num_prompts=num_prompts,
        min_input_len=1000,
        max_input_len=1001,
        min_output_len=10,
        max_output_len=20,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = get_dataset(dataset_args, tokenizer=tokenizer)
    for idx, req in enumerate(dataset):
        req.custom_params["created_time"] = idx
    return dataset


def test_pd_disagg_multi_instance():
    model_path = "Qwen/Qwen3-8B"
    prefill_workers, decode_workers = _create_pd_workers(
        model_path, num_prefill=2, num_decode=1
    )

    runner = PDDisaggBenchmarkRunner(
        prefill_workers=prefill_workers,
        decode_workers=decode_workers,
        prefill_lb_proxy=RoundRobinPolicy(),
        decode_lb_proxy=RoundRobinPolicy(),
    )

    dataset = _create_dataset(model_path, num_prompts=len(decode_workers) * 10)
    benchmark_config = BenchmarkConfig(ignore_request_timestamp=False)

    metrics = runner.benchmark(benchmark_config, dataset=dataset)
    assert metrics["completed"] == len(dataset)

    request_stats = runner.get_request_stats()

    for req in request_stats:
        for lat in req["gen_token_latencies"]:
            # Check token latency, specifically for the second token, 
            # which is the first token produced by the decoding worker.
            assert lat > 0

    runner.shutdown()


if __name__ == "__main__":
    test_pd_disagg_multi_instance()
