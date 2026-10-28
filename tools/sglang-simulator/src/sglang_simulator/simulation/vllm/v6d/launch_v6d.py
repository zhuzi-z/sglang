"""Launch v6d with daemon-side CPU simulation hooks installed first.

Usage mirrors the native CLI::

    python -m sglang_simulator.simulation.vllm.v6d.launch_v6d serve ...

Import ordering is intentional: v6d target classes must not be imported before
``install_v6d_runtime_hooks`` registers the class hooks.
"""

import sys

from sglang_simulator.simulation.vllm.v6d.v6d_capacity import (
    install_v6d_runtime_hooks,
)


def main() -> int:
    install_v6d_runtime_hooks()

    # Import only after hook registration.
    from v6d.cli.cli import main as v6d_main
    from sglang_simulator.hook.class_hook_entry import emit_hook_report
    emit_hook_report("daemon-cli-imported", "v6d-daemon")

    sys.argv[0] = sys.argv[0].removesuffix(".exe")
    return v6d_main()


if __name__ == "__main__":
    raise SystemExit(main())
