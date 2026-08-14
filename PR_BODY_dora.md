# Integrate the DoRA codec (weight-decomposed low-rank adaptation)

Adds `dora` as a registered generated-adapter shape, following the CONTRIBUTING.md
extension path. One codec per PR; the shared substrate (conditioner, trunk, training loop,
data, evaluator, controls) is untouched.

## The shape

DoRA (Liu et al., 2024) splits the frozen projection into a per-output-channel **magnitude**
and a **direction**, adapts the direction with LoRA-style factors, and renormalizes:

```
W' = m ⊙ (W0 + scale·B@A) / ||W0 + scale·B@A||_row
```

Both parts are generated: the emitted vector is LoRA's rank-8 `A`/`B` slices followed by
`out_features` magnitude scalars, which are a **delta on the frozen row norms**
(`m = ||W0||_row + scale·g_m`). So the zero-initialized head is exactly the frozen
projection, bit for bit, and `initial_bias()` seeds `A` only — the same bilinear
saddle-escape LoRA and LoKr use.

Hook sites are LoRA's: `q_proj`/`v_proj` for T2A, `down_proj` for D2A. Scale is swept with
the generic `--codec-scaling` flag; no new per-codec flag.

**Why this shape is interesting here.** Every other registered codec emits a self-contained
function of its generated values, so the hypernetwork has to discover the scale of a useful
edit from data. DoRA is the only one defined *relative to the weight it edits*: magnitude
and direction are separately addressable and pre-normalized by the host weight. It is
therefore a direct test of whether one-shot adapter prediction is limited by *calibration*
rather than expressivity — and of whether the LoRA-vs-full-finetuning deficit DoRA was
introduced to close has any counterpart when the adapter is predicted instead of optimized.

## One additive interface seam: `apply_at`

DoRA needs `W0`, and `base_output = W0 @ x` cannot recover it. So `GeneratedUpdateCodec`
gains

```python
def apply_at(self, module, inputs, base_output, generated, layer_index):
    return self.apply(inputs, base_output, generated, layer_index)   # default
```

and both forward-hook implementations (`hypernetwork.py::_apply_codec_hooks` and the
persistent-hook variant in `scripts/t2a_train_ddp.py`) now pass the resolved hook-site
module. Consequences, deliberately narrow:

- **Existing codecs are unchanged bit for bit** — the default ignores `module` and calls
  `apply()`.
- The frozen weight is **read-only**: it never enters the generated state, the optimizer, or
  `output_size`, and the interpreter stays frozen (a test asserts no gradient reaches it).
- A weight-decomposed codec must hook a **linear projection**, never the `"block"` residual
  stream. The codec enforces this with a shape check and a clear error.

## Faithfulness and cost

- **The denominator is detached**, exactly as in the reference implementation and PEFT's
  `DoraLinearLayer`: the paper treats `||V + ΔV||_c` as constant in the backward pass. The
  numerator `m` stays differentiable, so gradient reaches the magnitude slice.
- **No dense ΔW in the live path.** A per-example `(out, in)` tensor at every layer would
  dominate the step, so the row norms use the exact expansion
  `||W0 + s·BA||²_row = ||W0||²_row + 2s·⟨W0, BA⟩_row + s²·||BA||²_row`, which needs only
  `A@W0ᵀ` and the `r × r` Gram matrix `A@Aᵀ`. A test pins this against a materialized
  reference implementation, so the shortcut is verified rather than asserted.

## Budget (declared, not hidden)

DoRA's scalar count is the locked rank-8 LoRA budget **plus `d_out`** for the magnitude
vector: +2,048 on gemma-2-2b `q_proj` (+5.9%) and +1,024 on Qwen3-0.6B `down_proj` (+8.7%).
The magnitude vector is the method's shape identity, not a tunable extra; it is recorded in
`configs/adapters/dora.yaml` and should be read alongside any comparison with the
budget-matched shapes (FourierFT matches LoRA's budget exactly; DoRA cannot without ceasing
to be DoRA).

## Tests

`uv run pytest -q` → **144 passed**. New DoRA coverage: geometry and the budget rule at both
settings' locked projections; identity at initialization; the bilinear saddle-escaping bias
(gradient reaches the zero `B` slice *and* the magnitude slice); equivalence with the
materialized weight-decomposition reference; `dense_delta`/`apply` consistency;
magnitude-only channel rescaling; the required-weight and wrong-hook-site errors; hook
application and backprop through the hypernetwork with the interpreter frozen; and the
`StaticAdapter` (`matched − static*`) control path.

Also green: `adapterbench validate` (setups=2 adapters=6 trials=12), `adapterbench catalog`,
`adapterbench results validate`, `adapterbench results check` (no leaderboard drift),
`adapterbench website build`.

## GPU verification (plumbing, not a result)

| path | command | outcome |
|---|---|---|
| T2A single-GPU | `adapterbench t2a-sft --adapter dora --steps 60` | loss 7.41 → 1.17 |
| T2A 4-GPU DDP at the released effective batch (4 × 16) | `t2a_base_diag.sh google/gemma-2-2b-it … ADAPTER=dora` | 60 steps, 1.02 s/step, snapshot written |
| D2A document-conditioned | `adapterbench d2a-niah --adapters dora --steps 20 …` | trains, ctxswap control emitted |

## Leaderboard rows: in flight

Both rows are **pending** — no number is claimed in this PR. The AUTORESEARCH.md protocol is
running now, with ladders declared up front in append-only ledgers
(`results/autoresearch/{t2a,d2a}/dora/state.jsonl`, scratch):

- **Ladders** are centred on DoRA's *own* identity convention (scale 1.0), ×4 spacing over
  0.0625–16, safety limits 1/256 and 256 — never imported from another codec's selected
  scale (in particular not LoRA's D2A parity scale of ~45).
- **T2A**: scout seed 6702, selection seeds 6711–6713, confirmation seeds 6741–6743; both
  roles sweep the same ladder because the static* control is selected independently on its
  own selection-split ROUGE-L (`--example-offset 40`); the report split is spent exactly
  once.
- **D2A**: locked numeric-decoy substrate at 512 training tokens, 8k/16k/32k promotion rungs,
  dev eval seed 1802 at 12 examples/bin for selection; the held-out eval seed 2904 at 32
  examples/bin is reserved for the five confirmation seeds.

Search tooling ships with this PR (`scripts/t2a_dora_trial.sh`,
`scripts/t2a_dora_scale_locator.sh`, `scripts/d2a_dora_scale_locator.sh`). The
`scripts/reproduce/` wrappers, `scripts/reproduce/t2a_codec_config.sh` entry,
`canonical_results/*dora*.json`, and the two leaderboard rows land in a follow-up commit on
this branch once the confirmation seeds finish — including if the measured result is
negative, per the benchmark's policy that a shape which fails to condition is still a row.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
