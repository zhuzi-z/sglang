#!/usr/bin/env python3
"""Send an explicit kv_transfer_params probe to DashServing.

This script deliberately does not encode behavior in request_id.  Remote
control-plane behavior is enabled only by explicit kv_transfer_params supplied
through SamplingParams.extra and/or extra_params.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from dashserving.client import LLMClientV1, SamplingParams
from transformers import AutoTokenizer


def _json_obj(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe native V6D control-plane with explicit kv params."
    )
    parser.add_argument("--endpoint", default="127.0.0.1:8001")
    parser.add_argument("--request-id", default=f"qoder-kv-probe-{int(time.time())}")
    parser.add_argument("--model", default="/root/workspace/models/Qwen/Qwen3___5-0___8B")
    parser.add_argument("--repeat", type=int, default=187)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument(
        "--kv",
        type=_json_obj,
        default={"do_remote_decode": True},
        help="JSON object for kv_transfer_params",
    )
    parser.add_argument(
        "--extra",
        type=_json_obj,
        default={},
        help="Additional extra_params JSON object",
    )
    parser.add_argument(
        "--extra-params",
        action="store_true",
        help="Also send kv_transfer_params through LLMClient extra_params",
    )
    args = parser.parse_args()

    extra_params = dict(args.extra)
    if args.extra_params:
        extra_params["kv_transfer_params"] = args.kv

    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        extra={"kv_transfer_params": args.kv},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text = ("跨机前缀匹配验证。请保持这个公共前缀完全一致。" * args.repeat) + "\n问题：请用一个字回答。"
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    print("request_id", args.request_id, flush=True)
    print("tokens", len(token_ids), flush=True)
    print("kv_transfer_params", args.kv, flush=True)
    print("extra_params", extra_params or None, flush=True)

    client = LLMClientV1(args.endpoint)
    for response in client.generate(
        request_id=args.request_id,
        prompt={"prompt_token_ids": token_ids},
        params=params,
        extra_params=extra_params or None,
    ):
        print("resp", response, flush=True)


if __name__ == "__main__":
    main()
