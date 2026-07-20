"""Summarize completed selective-erasure scale probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()

    header = (
        "scale  matched-static (SE)  matched-swapped (SE)  "
        "erase-suppress  retain-change  CLIP-T-change  decision"
    )
    print(header)
    for run in args.runs:
        result_path = run / "probe_results.json" if run.is_dir() else run
        payload = json.loads(result_path.read_text())
        config, metrics = payload["config"], payload["metrics"]
        print(
            f"{config['scale']:>5g}  "
            f"{metrics['matched_minus_static']:+.5f} "
            f"({metrics['matched_minus_static_se']:.5f})  "
            f"{metrics['matched_minus_swapped']:+.5f} "
            f"({metrics['matched_minus_swapped_se']:.5f})  "
            f"{metrics['erase_suppression_vs_frozen']:+.5f}       "
            f"{metrics['retain_change_vs_frozen']:+.5f}       "
            f"{metrics['clipt_change_vs_frozen']:+.5f}       "
            f"{'GO' if metrics['go'] else 'NO-GO'}"
        )


if __name__ == "__main__":
    main()
