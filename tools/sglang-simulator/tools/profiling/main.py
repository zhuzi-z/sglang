import json
import os
from argparse import ArgumentParser
from pathlib import Path

import sgl_profile


def load_request_tokens_from_jsonl(file: Path) -> dict[str, dict]:
    data = {}
    with file.open() as f:
        line = f.readline()
        while line:
            req = json.loads(line)
            data[req["rid"]] = req
            line = f.readline()
    return data


def load_schedule_batch_from_jsonl(file: Path) -> list[dict]:
    data = []
    with file.open() as f:
        line = f.readline()
        while line:
            req = json.loads(line)
            data.append(req)
            line = f.readline()
    return data


def build_profile_batch_list(
    schedule_batch_file: str,
    request_tokens_file: str | None = None,
    schedule_row_number_filter: list[int] | None = None,
) -> list[sgl_profile.ScheduleBatchRequest]:
    schedule_batch_file: Path = Path(schedule_batch_file)
    if not schedule_batch_file.exists():
        raise FileExistsError(schedule_batch_file)

    schedule_batch_list = load_schedule_batch_from_jsonl(schedule_batch_file)

    if request_tokens_file is not None:
        request_tokens_file: Path = Path(request_tokens_file)
        if not request_tokens_file.exists():
            raise FileExistsError(f"{request_tokens_file}")
        request_tokens_dict = load_request_tokens_from_jsonl(request_tokens_file)

    if schedule_row_number_filter is None:
        schedule_row_number_filter = list(range(len(schedule_batch_list)))

    selected_batch_list = []
    for row_id in schedule_row_number_filter:
        request_infos = schedule_batch_list[row_id]["request_infos"]
        batch = sgl_profile.ScheduleBatch([])
        for req in request_infos:
            if request_tokens_file is not None:
                req_tokens: dict = request_tokens_dict.get(req["rid"])
                if req_tokens is None:
                    print(
                        f"Fail to get request detail token ids from {request_tokens_file}"
                    )
                    input_ids, output_ids = None, None
                else:
                    input_ids = req_tokens.get("input_ids")
                    output_ids = req_tokens.get("output_ids")
            batch.reqs.append(
                sgl_profile.ScheduleBatchRequest(
                    extend_len=req["extend_input_len"],
                    past_kv_len=req["prefix_indices_len"] + req["output_ids_len"],
                    input_ids=input_ids,
                    output_ids=output_ids,
                )
            )
        selected_batch_list.append(batch)

    return selected_batch_list


def main():
    parser = ArgumentParser()
    parser.add_argument("--server-args", type=str, required=True)
    parser.add_argument("--schedule-batch-file", type=str, required=True)
    parser.add_argument("--request-tokens-file", type=str)
    parser.add_argument("--schedule-row-number-filter", type=int, nargs="+")
    parser.add_argument(
        "--profiler", default="torch", type=str, choices=["nsys", "torch"]
    )
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--num-replay", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=2)

    args = parser.parse_args()

    from sglang.srt.server_args import ServerArgs

    server_args = ServerArgs(**json.loads(args.server_args))
    server_args.disable_overlap_schedule = True

    batch_list = build_profile_batch_list(
        args.schedule_batch_file,
        args.request_tokens_file,
        args.schedule_row_number_filter,
    )

    sgl_profile.run(
        server_args=server_args,
        batch_list=batch_list,
        output_dir=args.output_dir or os.getcwd(),
        profiler=args.profiler,
        num_replay=args.num_replay,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
