"""Launch the VoiceForge speech application.

Examples
--------
Start the interface on the default port::

    python app.py

Use a different checkpoint and open the browser automatically::

    python app.py --checkpoint checkpoints/voiceforge_attn.pt --open

The interface runs entirely locally: no audio or text leaves the machine.
"""

from __future__ import annotations

import argparse
import sys

from voiceforge.app_core import DEFAULT_CHECKPOINT, DEFAULT_OUTPUT_DIR, AppError, VoiceForgeApp
from voiceforge.web_app import run

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the VoiceForge text-to-speech application.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Compute device. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=32,
        help="Griffin-Lim iterations (quality versus speed).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the interface in the default browser.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not 1 <= args.port <= 65535:
        print("ERROR: --port must be between 1 and 65535", file=sys.stderr)
        return 1

    app = VoiceForgeApp(
        checkpoint_path=args.checkpoint,
        device=args.device,
        output_dir=args.output_dir,
        griffin_lim_iterations=args.n_iter,
    )

    # Load once at startup so a missing or broken checkpoint is reported
    # immediately instead of on the first button click.
    try:
        app.load()
    except AppError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {app.device_description()}")

    run(
        app,
        host=args.host,
        port=args.port,
        open_browser=args.open,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
