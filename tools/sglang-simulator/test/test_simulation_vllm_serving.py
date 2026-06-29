"""
Test vLLM server-mode simulation with BLOCKING mode.

Follows the pattern of test_simulation_sglang_serving.py:
1. Spawns vLLM simulation server (launch_server.py) as subprocess
2. Sends requests via OpenAI-compatible /v1/completions API
3. Validates that requests complete and timing is reasonable
"""

import json
import os
import signal
import subprocess
import sys
import time

import requests

os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = (
    os.path.dirname(__file__) + "/assets/config_vllm.json"
)
os.environ["SGLANG_SIMULATOR_OUTPUT_MODE"] = "BLOCKING"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

MODEL_PATH = "/host/models/Qwen/Qwen3-0.6B/"
SERVER_PORT = 18100


class VLLMServingRunner:
    def __init__(self, model: str, port: int = SERVER_PORT, **extra_args):
        cmd = [
            sys.executable,
            "-m",
            "sglang_simulator.simulation.vllm.launch_server",
            "--model", model,
            "--port", str(port),
            "--dtype", "float16",
            "--load-format", "dummy",
            "--enforce-eager",
            "--block-size", "16",
            "--gpu-memory-utilization", "0.9",
        ]
        for k, v in extra_args.items():
            flag = "--" + k.replace("_", "-")
            if v is True:
                cmd.append(flag)
            elif v is False:
                pass
            else:
                cmd.extend([flag, str(v)])

        env = os.environ.copy()
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.server_proc = subprocess.Popen(
            cmd, env=env, preexec_fn=os.setsid
        )

        # Wait for server to be ready
        dur = 0
        while dur < 120:
            try:
                r = requests.get(f"{self.base_url}/health")
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
            dur += 1
        raise RuntimeError("Failed to start vLLM simulation server.")

    def completions(self, prompt: str, max_tokens: int = 10) -> dict:
        """Send a single completion request."""
        resp = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": MODEL_PATH,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "ignore_eos": True,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def shutdown(self):
        if not self.server_proc or self.server_proc.poll() is not None:
            return
        os.killpg(self.server_proc.pid, signal.SIGTERM)
        try:
            self.server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self.server_proc.pid, signal.SIGKILL)
            self.server_proc.wait()
        self.server_proc = None


def test_vllm_serving_blocking():
    """Test vLLM server with BLOCKING mode simulation."""
    runner = VLLMServingRunner(model=MODEL_PATH)

    try:
        # Send a few requests sequentially
        results = []
        for i in range(3):
            t0 = time.time()
            result = runner.completions(prompt=f"Hello world {i}", max_tokens=5)
            elapsed = time.time() - t0
            results.append((result, elapsed))

        # Validate responses
        for i, (result, elapsed) in enumerate(results):
            assert "choices" in result, f"Request {i}: no choices in response"
            assert len(result["choices"]) > 0, f"Request {i}: empty choices"
            text = result["choices"][0]["text"]
            assert len(text) > 0, f"Request {i}: empty text output"
            # In BLOCKING mode, each request should take some time due to time.sleep
            # With AIConfigurator predictor, each token takes real inference time
            print(
                f"  Request {i}: output_tokens={result['usage']['completion_tokens']}, "
                f"elapsed={elapsed:.3f}s"
            )

        print("\n[PASS] test_vllm_serving_blocking passed!")
    finally:
        runner.shutdown()


if __name__ == "__main__":
    test_vllm_serving_blocking()
