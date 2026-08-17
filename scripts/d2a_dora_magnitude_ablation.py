#!/usr/bin/env python3
"""DIAGNOSTIC: run the D2A NIAH setting with DoRA's generated MAGNITUDE channel disabled.

Not a benchmark result and not publishable as a DoRA row: switching off part of a codec's
shape identity produces a different shape. Its only job is to answer *why* DoRA fails to
retrieve, if it does — a question the leaderboard row should be able to state a mechanism
for rather than leaving as "it just doesn't".

The mechanism under test: DoRA gives the hypernetwork two routes to change the hooked
projection — a rank-8 direction and a per-output-channel magnitude. The magnitude route is
low-dimensional and high-gain (it rescales whole output channels), so it can plausibly
dominate the optimisation, driving the language-model loss down while crowding out the
low-rank route that document-conditioned retrieval actually needs. Every DoRA D2A run so
far shows exactly that signature: loss collapsing to ~0.00 with gate retrieval pinned at
0.000. If disabling the magnitude channel restores retrieval, that signature has a cause;
if it does not, the failure is not about the magnitude channel at all. Either answer is
worth having, and neither is a leaderboard number.

Implementation: the generated magnitude slice is forced to exactly zero for the whole run,
so `m = ||W0||_row` identically and the update reduces to a renormalised low-rank one. This
is done by zeroing the magnitude rows of each per-module head (weight AND bias) and
registering gradient hooks that keep them zero, so the optimiser can never move them. The
codec itself is untouched — no benchmark semantics change, and the frozen interpreter, the
conditioner, the data, the hook site, and the evaluator are all exactly the shipped ones,
reached through the real `d2a-niah` command path.

Usage: same flags as `adapterbench d2a-niah` (they are forwarded verbatim), e.g.
  d2a_dora_magnitude_ablation.py --adapters dora --codec-scaling 4 \
      --needle-style realistic_numeric_decoys --numeric-decoy-count 4 --context-lengths 512 \
      --eval-context-lengths 512,1024 --num-train-documents 512 --steps 8000 --eval-every 4000 \
      --learning-rate 2e-5 --warmup-steps 960 --n-latents 208 --num-blocks 8 --eval-limit 12 \
      --eval-seed 1802 --seed 902 --device cuda:4 --output <dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from adapterbench.cli import build_parser  # noqa: E402
from adapterbench.t2a.codecs import DoRACodec  # noqa: E402
from adapterbench.t2a.hypernetwork import TextToPeftHypernetwork  # noqa: E402


def _freeze_magnitude_slice(hypernetwork: TextToPeftHypernetwork) -> int:
    """Zero and pin the magnitude rows of every DoRA head. Returns how many were pinned."""
    pinned = 0
    for name, codec in hypernetwork.codecs.items():
        if not isinstance(codec, DoRACodec):
            continue
        start = codec.rank * (codec.in_features + codec.out_features)
        head = hypernetwork.heads[name]
        with torch.no_grad():
            head.weight[start:].zero_()
            head.bias[start:].zero_()

        def zero_rows(grad, start=start):
            grad = grad.clone()
            grad[start:] = 0
            return grad

        head.weight.register_hook(zero_rows)
        head.bias.register_hook(zero_rows)
        pinned += head.weight.shape[0] - start
        print(f"[ablation] {name}: magnitude slice [{start}:{head.weight.shape[0]}] zeroed and pinned",
              flush=True)
    return pinned


def main() -> int:
    original_init = TextToPeftHypernetwork.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not _freeze_magnitude_slice(self):
            raise SystemExit("no DoRA codec found: this diagnostic only applies to --adapters dora")

    TextToPeftHypernetwork.__init__ = patched_init
    print("[ablation] DoRA magnitude channel DISABLED (m = ||W0||_row identically). "
          "Diagnostic only — not a benchmark result.", flush=True)

    parser = build_parser()
    parsed = parser.parse_args()
    if parsed.command != "d2a-niah":
        raise SystemExit("this diagnostic wraps the d2a-niah command only")
    parsed.func(parsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
