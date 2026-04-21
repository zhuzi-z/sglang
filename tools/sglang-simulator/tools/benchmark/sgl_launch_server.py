import argparse
import os

from sgl_hook import C_SglangSchedulerReqHook, C_TokenizerManagerHook
from sglang_simulator.hook import install_class_hooks

install_class_hooks([C_SglangSchedulerReqHook, C_TokenizerManagerHook])


# Ref: https://github.com/sgl-project/sglang/blob/v0.5.6.post2/python/sglang/launch_server.py
if __name__ == "__main__":
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree

    parser = argparse.ArgumentParser()

    server_args = ServerArgs.from_cli_args(parser.parse_args())

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
