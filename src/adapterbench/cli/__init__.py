"""AdapterBench CLI.

The command implementations live in per-topic modules; this file only builds the
argument parser and dispatches. Console entry point is `adapterbench.cli:main`.

- meta.py          catalog / validate / matrix / doctor / peft-smoke
- live_sft.py      t2a-sft / t2a-sft-pilot / d2a-niah / t2a-sft-sweep
"""

from __future__ import annotations

import argparse

from . import live_sft, meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark PEFT adapters as hypernetwork outputs")
    parser.set_defaults(func=lambda _: parser.print_help())
    subparsers = parser.add_subparsers(dest="command")
    meta.register(subparsers)
    live_sft.register(subparsers)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
