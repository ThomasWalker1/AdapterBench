# AdapterBench: Project Plan

## Status snapshot (as of 2026-07-05)

Moved to `/home/tw78/AdapterBench` (renamed again from an intermediate
`/home/tw78/peft_for_hnets` — each move breaks every installed console script's shebang,
since those are absolute paths baked in at install time; fix is always
`uv pip install --python .venv/bin/python --reinstall -e ".[dev]"` to regenerate them,
not a fresh `uv venv`) and put under git version control (repo:
`https://github.com/ThomasWalker1/AdapterBench`, private) — rebranded from its earlier
working name ("PEFT-as-Hypernetwork-Output Benchmark" / `peft-hnet-benchmark`) to
**AdapterBench**. The installable Python package (`peft_hnet`) and CLI command
(`peft-hnet`) keep their existing names to avoid an invasive import-path rename; only the
repo/distribution name and docs are rebranded. `archive/` (retired Phase 3 code) was
removed from the working tree once git history existed to fall back on — see the repo's
first two commits, not a live directory.

**Phase 4: the benchmark has been restructured to train solely via live end-to-end SFT**
— the hypernetwork's generated output is hooked directly into a real frozen interpreter's
forward pass on real training examples, computing ordinary next-token cross-entropy loss
and backpropagating through the hook into the hypernetwork. This is Sakana's own actual
main training method (not the reconstruction-matching approach Phase 3 built), and it's
a strictly better foundation for the project's real goal — an "almost plug-and-play"
mechanism for testing *any* new PEFT representation, weight-based or activation-based,
without needing a representation-specific oracle-target derivation step at all. A new
representation now needs exactly two things: an output structure (a codec) and a hook
site (a named linear submodule, or `"block"` for the whole decoder layer/residual
stream) — the training loop, data pipeline, and evaluator are the same for every
representation. Proven end-to-end on a real model for both a weight-space representation
(LoRA) and an activation-space one (a new `ActivationSteeringCodec`) using the identical
CLI command, just swapping `--representation`/`--target-modules`. Then taken all the way
to a genuine three-way comparison: `peft-hnet t2p-sft-pilot` trains LoRA, IA3, and
activation steering independently on a shared 8-task split and scores each against real
held-out `boolq`/`hellaswag` examples — first honest result: IA3 and activation steering
(the two low-parameter-count representations) both beat the frozen baseline, while LoRA
(the largest, ~40x more generated parameters than steering) actively destroys downstream
performance, below-chance on `boolq`. See the Phase 4 section below for the full writeup
— including two real, previously-undiscovered bugs this exposed: LoRA/LoKr's dead
zero-gradient saddle point at this hypernetwork's default init (fixed at the codec
level), and Qwen3's default "thinking" chat-template mode silently breaking both training
and eval prompts unless `enable_thinking=False` is passed explicitly.

**Phase 3's reconstruction-matching pilot is retired** now that live SFT is the sole
training/eval mechanism; its code was removed from the working tree once this project
gained git history (recoverable via `git log`/`git show` on the repo's first two
commits, not a live `archive/` directory), and its real results and findings are
preserved below for history.
One old finding from even before Phase 3 (an unresolved KronA initialization issue, see
just below the Phase 2 results) turned out to be the same *family* of problem as gotcha
#15's LoRA/LoKr fix here — bilinear factorized codecs need deliberate asymmetric
initialization, a recurring theme worth remembering for any future bilinear codec.

Phase 1 (Text-to-LoRA reconstruction setting, released checkpoint, LoRA representation
only) is implemented and validated end-to-end on real hardware and real data — this is
the first genuine result the benchmark has produced. A full run (5 tasks x 100 examples,
`lora` + `frozen_interpreter` arms) completed at
`results/text_to_peft_gemma2b_reconstruction_phase1/results.jsonl`:

| task | lora (T2L) | frozen_interpreter |
|---|---|---|
| arc_easy | 0.71 | 0.59 |
| arc_challenge | 0.43 | 0.44 |
| boolq | 0.81 | 0.83 |
| hellaswag | 0.55 | 0.48 |
| gsm8k (exact_match) | 0.13 | 0.57 |

**Notable finding (verified, not a scoring bug — manually inspected raw generations
before trusting it):** the released T2L LoRA for gsm8k causes Gemma-2-2b-it to emit a
bare terse number ("8", "3", "$45,000") with no reasoning, whereas the frozen model
naturally does step-by-step chain-of-thought and gets most of them right. The LoRA
adapter is actively suppressing the base model's existing CoT behavior — plausibly an
artifact of the short-QA-style (Lots-of-LoRAs) task distribution it was conditioned on —
and Gemma-2-2b-it apparently can't do this arithmetic correctly without reasoning through
it. Net effect: LoRA helps or ties on the four classification-style tasks but is sharply
*worse* than doing nothing on the one task requiring multi-step reasoning. Relevant for
Phase 3: a representation can look "worse" not because it lacks capacity, but because it
overwrites a capability the base model already had on tasks outside its conditioning
distribution — a different failure mode than raw expressivity, worth measuring
separately (e.g. compare adapted-vs-frozen behavior on tasks *unrelated* to what the
adapter was generated for) once multiple representations are being compared head to head.

**Phase 2 (train our own hypernetwork checkpoint, still LoRA-only) is also complete.**
Trained 8 real oracle per-task LoRAs for `mistralai/Mistral-7B-Instruct-v0.2` (tasks
`lol_022, lol_033, lol_034, lol_035, lol_039, lol_043, lol_044, lol_045`, via upstream's
`scripts/train_lora_baselines.py`, saved under
`upstream/text-to-lora/train_outputs/sft/oracle_lora/*/`), then trained our own
reconstruction hypernetwork against them (`scripts/train_hyper_recon.py`, 10,000 epochs,
converged — recon loss flat around 0.156 since roughly epoch 6000), producing
`upstream/text-to-lora/train_outputs/recon/hyper_lora/20260705-114020_VmdhLFWM/hypermod.pt`.
Plugged that checkpoint into the *same* `ReleasedTextToLoRABackend`/`HFDownstreamEvaluator`
Phase 1 built (new setup manifest:
`configs/setups/text_to_peft_mistral7b_reconstruction_pilot.yaml`) and confirmed it
generalizes beyond the one released checkpoint it was originally built against — real,
non-degenerate results on two held-out benchmarks the checkpoint never saw during its
narrow 8-task training:

| task | lora (self-trained, 8-task pilot) | frozen_interpreter |
|---|---|---|
| arc_easy (n=20) | 0.70 | 0.65 |
| boolq (n=20) | 0.85 | 0.85 |

Small n, so don't read much into the arc_easy gap — the point was proving the harness
works against a self-trained checkpoint, which it does. See gotchas #7-10 below for real
problems hit and fixed/worked around along the way (a `--model_dir` trailing-slash bug I
introduced, two separate non-fatal crashes in upstream's own automatic eval steps, and an
unreliable background-task-completion signal on this machine).

Environment: a dedicated `uv` venv at the repo root (`.venv/` + committed `uv.lock`) —
not any pre-existing conda env. `google/gemma-2-2b-it` is gated; needs `hf auth login`
with license acceptance on the same HF account (see SETUP.md).

Prior planning docs and an early CPU-only synthetic diagnostic pilot (`archive/`) have
been deleted — superseded by the real implementation described below. One durable
finding from that pilot is worth keeping in mind for Phase 3: KronA's compiled error
stayed high (0.981) despite near-zero oracle error on held-out Kronecker-structured
targets — i.e. the representation could express the target perfectly, but the
hypernetwork's mapper failed to learn to generate the right nonlinear factors. That
motivates factor-aware initialization or a delta-space auxiliary loss whenever Phase 3
gets to KronA, rather than assuming a naive basis-mixing head will "just work" for it.

## Architecture (what actually exists)

- `src/peft_hnet/contracts.py` — the core abstraction everything below implements:
  `HypernetworkBackend.generate(conditions, output_dir) -> Mapping[str, AdapterArtifact]`,
  `DownstreamEvaluator.evaluate(artifacts, examples, split) -> list[EvaluationResult]`.
- `src/peft_hnet/text_to_lora_backend.py::ReleasedTextToLoRABackend` — the first concrete
  `HypernetworkBackend`. Wraps a released T2L `hypermod.pt` checkpoint. Generation runs
  out-of-process (`scripts/generate_t2l_adapter.py`) under `upstream/text-to-lora/.venv`,
  because `hyper_llm_modulator` pins a torch/transformers/peft stack incompatible with
  `peft_hnet`'s own — **never import it directly from `peft_hnet`.**
- `src/peft_hnet/hf_downstream_evaluator.py::HFDownstreamEvaluator` — the first concrete
  `DownstreamEvaluator`. Plain `transformers`+`peft` (no vLLM — not installed, and
  non-LoRA representations this benchmark ultimately compares aren't vLLM-servable
  anyway). Replicates T2L's exact tokenizer/chat-template setup so scoring is comparable
  to upstream. Has `iter_evaluate`/`iter_evaluate_frozen` generator methods (yield one
  result per task as it completes) — always prefer these over the batch
  `evaluate`/`evaluate_frozen` for anything that takes more than a minute or so, so a
  crash/kill doesn't lose all progress (see "Lessons learned" below).
- `src/peft_hnet/task_examples.py` — builds `TaskExample`s for
  arc_easy/arc_challenge/boolq/hellaswag/gsm8k from public HF datasets; condition text
  is sourced from the checkpoint's own `args.yaml::eval_ds_info`.
- `src/peft_hnet/t2p/codecs.py` — representation-agnostic differentiable codecs, each
  hookable at either a named linear submodule (weight-space: LoRA, FreezeALoRA, LoKr,
  FourierFT) or a whole decoder layer's output/residual stream (activation-space: IA3,
  `ActivationSteeringCodec`). Base class is `GeneratedUpdateCodec` (renamed from
  `LinearUpdateCodec` — no longer linear-submodule-specific). Every codec except IA3/
  activation steering (no weight-space delta) exposes `dense_delta(generated,
  layer_index) -> (batch, out, in)` — a holdover from the retired reconstruction path,
  kept because it's cheap and harmless, not used by live SFT. `initial_bias()` lets a
  codec override the hypernetwork's default all-zero head init — LoRA/LoKr must (see
  gotcha #15); everything else is fine at zero.
- `src/peft_hnet/t2p/hypernetwork.py::TextToPeftHypernetwork` — `apply()`'s hook closure
  now handles both bare-tensor (linear submodule) and tuple (decoder layer) outputs;
  `_resolve_target` resolves `"block"` to the layer itself as a sentinel (matching
  upstream Sakana's own `hooks.py` convention) and anything else to a named submodule;
  `infer_module_shapes` accepts an optional `hidden_size` for non-`nn.Linear` targets.
  `forward_layer(condition_embeddings, layer_index)` (a Phase 3 holdover, memory-saving
  for dense-ΔW reconstruction training) is unused by live SFT but left in place.
- `src/peft_hnet/t2p/model_utils.py::get_decoder_layers` — resolves a live
  `AutoModelForCausalLM`'s decoder-layer list (port of upstream's `get_layers`).
- `src/peft_hnet/t2p/lol_data.py` — Super-NaturalInstructions training data via its
  per-task `Lots-of-LoRAs/task*` HF Hub mirror (confirmed the only SNI access path
  Sakana's own repo uses — no raw SNI dataset reference exists anywhere upstream).
  `load_task_metadata` reads the already-cloned `upstream/text-to-lora/tasks/*/
  metadata.yaml`; `preprocess_lol_example` ports the exact `Definition:`/`Now complete
  the following example -`/`Output:` string-split convention; `tokenize_prompt_response`
  does response-only label masking (`-100` on prompt tokens) via a simplified,
  tokenizer-portable length-based split rather than upstream's sequence-pair
  `sequence_ids()` approach; `LolSFTDataset`/`lol_collate_fn` build batches. Points at
  the existing per-task descriptions as-is (a small prefix, not the paper's full 128) —
  see the `# TODO` in the module docstring re: unresolved GPT-4o-mini provenance.
- `src/peft_hnet/t2p/sft_trainer.py` — the live training loop itself.
  `compute_sft_loss` runs `hypernetwork(condition_embeddings)` then
  `hypernetwork.apply(layers, generated)` around a real interpreter forward pass,
  `masked_cross_entropy` does upstream's per-example (not per-token) loss averaging
  plus optional label smoothing, `train_step`/`train_downstream_hypernetwork` optimize
  `hypernetwork.parameters()` only (interpreter frozen). No oracle adapters anywhere in
  this path.
- `src/peft_hnet/t2p/live_evaluator.py::HypernetworkDownstreamEvaluator` — the downstream
  evaluator counterpart: activates a representation via `hypernetwork.apply(...)` (same
  mechanism training used) instead of `HFDownstreamEvaluator`'s `peft.PeftModel.
  load_adapter`/`set_adapter`, so hook-based representations (activation steering, or any
  representation trained via `sft_trainer.py`) can be scored at all — `HFDownstreamEvaluator`
  fundamentally cannot load them, since they were never materialized as PEFT adapters.
  Its scoring routines (`_score_multiple_choice`, `_score_gsm8k`) are verbatim ports —
  representation-agnostic, only the activation mechanism differs.
- `src/peft_hnet/t2p/tiny_interpreter.py`, `t2p/synthetic_tasks.py`, `t2p/synthetic_evaluator.py`
  — the Phase-5 lightweight setting: a tiny freshly initialized `LlamaForCausalLM` (no
  download), 5 synthetic algorithmic task families with exactly known correct answers,
  and a function-based exact-match evaluator. Reuses `t2p/sft_trainer.py`,
  `t2p/codecs.py`, `t2p/hypernetwork.py`, `t2p/condition_encoder.py`, and even
  `t2p/lol_data.py::lol_collate_fn` completely unchanged — only the interpreter and data
  source differ from the real setting.
- `src/peft_hnet/cli.py` — `peft-hnet {catalog,validate,matrix,doctor,peft-smoke,run,t2p-sft,t2p-sft-pilot,t2p-synthetic-pilot}`.
  `run` is the Phase-1 entrypoint (unchanged: generate → evaluate → evaluate_frozen via
  the disk-artifact `HypernetworkBackend`/`DownstreamEvaluator` contract — still the
  right tool for evaluating Sakana's released/self-trained checkpoints as baselines).
  `t2p-sft` is the Phase-4 single-representation entrypoint: load interpreter → embed task
  descriptions → load + tokenize Lots-of-LoRAs examples → train the hypernetwork live →
  write loss-curve JSON. Replaces the retired `t2p-pilot` command. `t2p-sft-pilot` builds
  on the same pieces to train and compare *multiple* representations in one run (default
  `lora,ia3,activation_steering`, each with a fixed default hook site — see
  `_PILOT_DEFAULT_TARGET_MODULES`): shares one loaded interpreter/condition-encoder/
  training-batches across representations, trains each independently to convergence, then
  scores each (plus one shared `frozen_interpreter` baseline) via
  `HypernetworkDownstreamEvaluator` against real held-out benchmark examples, writing
  `results.jsonl`/`.csv` (via the existing `reporting.write_results`, reused unchanged)
  and a per-representation `loss_curves.json` incrementally after each representation
  completes (crash-safety, matching gotcha #3's established pattern). `t2p-synthetic-pilot`
  is the Phase-5 lightweight-setting entrypoint: same shared-then-loop structure, but
  builds a tiny from-scratch interpreter instead of downloading one, defaults to *all six*
  registered representations (affordable at this scale), and defaults `--device cpu`.
- `src/peft_hnet/reporting.py::write_results` — writes `results.jsonl` (schema-free,
  always safe) and `results.csv` (fieldnames are the *union* of metric keys across all
  rows — see gotcha #4 below).
- Environment: `uv venv .venv` + `uv pip install -e ".[dev]"` + committed `uv.lock` at
  the repo root, independent of any pre-existing conda env. `upstream/text-to-lora/.venv`
  is a second, separate `uv` env (older, upstream-pinned stack), gitignored, provisioned
  per `SETUP.md`.

## How to check current status / resume

```bash
cd /home/tw78/AdapterBench
uv run pytest -q                                                             # 68 tests as of 2026-07-05
cat results/text_to_peft_gemma2b_reconstruction_phase1/results.jsonl         # Phase-1 numbers: released T2L checkpoint, lora vs frozen_interpreter, 5 tasks x 100 examples
cat results/text_to_peft_mistral7b_reconstruction_pilot/results.jsonl        # Phase-2 numbers: our self-trained 8-task checkpoint, lora vs frozen_interpreter, 2 tasks x 20 examples
cat results/t2p_sft/smoke_lol022_lora.json                                  # Phase-4 smoke: real live-SFT loss curve, LoRA, 1 task, Qwen3-0.6B
cat results/t2p_sft/smoke_lol022_steering.json                              # Phase-4 smoke: same, activation_steering — same command, different --representation/--target-modules
cat results/t2p_sft_pilot/results.jsonl                                     # Phase-4 pilot: lora vs ia3 vs activation_steering vs frozen, boolq+hellaswag (n=20/family)
cat results/t2p_sft_pilot/loss_curves.json                                  # Phase-4 pilot: full 400-step loss curve per representation
cat results/t2p_synthetic_pilot_long/results.jsonl                          # Phase-5 pilot: all 6 representations vs frozen, 5 synthetic task families, 1500 steps
cat results/t2p_synthetic_pilot_long/loss_curves.json                       # Phase-5 pilot: full 1500-step loss curve per representation
git show 9afd7d3 -- archive/                                                 # retired Phase 3 reconstruction-matching code (removed from the working tree, recoverable via git history)
```

To rerun the Phase-4 multi-task pilot (needs a real GPU; trains 3 representations to 400
steps each, then scores each against real held-out `boolq`/`hellaswag` examples — takes a
few minutes on an idle A100 for `Qwen/Qwen3-0.6B`):

```bash
uv run peft-hnet t2p-sft-pilot --device cuda:0 --output results/t2p_sft_pilot
```

To rerun the Phase-5 lightweight synthetic pilot (no GPU or network needed; all 6
representations, ~3 min for 400 steps or ~20 min for 1500 steps on CPU):

```bash
uv run peft-hnet t2p-synthetic-pilot --steps 1500 --output results/t2p_synthetic_pilot_long
```

To rerun the Phase-4 live-SFT smoke test (needs a real GPU; loads a real interpreter —
unlike the retired Phase 3 pilot, this path always loads the interpreter model):

```bash
uv run peft-hnet t2p-sft --device cuda:0 --tasks lol_022 \
  --representation lora --target-modules q_proj,v_proj \
  --steps 60 --output results/t2p_sft/smoke_lol022_lora.json
# swap for activation steering with no other code changes:
uv run peft-hnet t2p-sft --device cuda:0 --tasks lol_022 \
  --representation activation_steering --target-modules block \
  --steps 60 --output results/t2p_sft/smoke_lol022_steering.json
```

If the Phase-1 results file is missing/stale/incomplete, rerun it — it's idempotent and
cheap (~10-20 min total, mostly gsm8k's 200 greedy-decode calls):

```bash
uv run peft-hnet run \
  --setup text_to_peft_gemma2b_reconstruction \
  --adapter lora_r8_t2l \
  --checkpoint upstream/text-to-lora/trained_t2l/gemma_2b_t2l/hypermod.pt \
  --tasks arc_easy,arc_challenge,boolq,hellaswag,gsm8k \
  --limit 100 \
  --output results/text_to_peft_gemma2b_reconstruction_phase1
```

Requires `hf auth login` (or `HF_TOKEN`) with license-accepted access to gated
`google/gemma-2-2b-it` (see SETUP.md).

## Lessons learned / gotchas (read before touching the pipeline again)

1. **Subprocess path resolution.** `ReleasedTextToLoRABackend` runs its generation
   subprocess with `cwd=upstream/text-to-lora`, so every path passed as a CLI arg must
   be pre-resolved to absolute — relative paths silently resolve against the wrong
   directory.
2. **Never `.resolve()` the upstream interpreter path.**
   `upstream/text-to-lora/.venv/bin/python` is a symlink; fully resolving it collapses
   to the real system interpreter and skips venv site-package activation
   (`hyper_llm_modulator` becomes `ModuleNotFoundError`). Only resolve *data* paths,
   never the interpreter path itself.
3. **Long-running steps must stream progress and persist incrementally.** The first
   full Phase-1 attempt was killed by the harness mid-evaluation with zero results
   saved, because the original code only printed/wrote output once at the very end.
   Fixed via the `iter_evaluate*` generators plus writing `results.jsonl`/`.csv` after
   every single task. Any new long-running command should follow this pattern from
   the start, not retrofit it after a loss.
4. **`reporting.write_results`'s CSV fieldnames must be a union across all rows**, not
   derived from the first row — different task families report different metric names
   (`accuracy` for multiple-choice tasks vs `exact_match` for gsm8k). Regression test:
   `tests/test_reporting.py`.
5. **`google/gemma-2-2b-it` is gated** — requires `hf auth login` (the modern `hf` CLI,
   not the deprecated `huggingface-cli`) plus accepting the license on the model's HF
   page with the same account.
6. Removed: the redundant top-level `text-to-lora/` clone (superseded by
   `upstream/text-to-lora/`, which already has an identical `chat_templates/`) and the
   entire `archive/` directory (superseded planning docs + a CPU-only synthetic
   diagnostic pilot; its one durable finding is captured above).
7. **`--model_dir` must not have a trailing slash** when passed to
   `scripts/train_hyper_recon.py` (or anything else that hands it to
   `AutoModelForCausalLM.from_pretrained`/`hf_hub_download`) — `huggingface_hub`'s
   `validate_repo_id` rejects `"mistralai/Mistral-7B-Instruct-v0.2/"` outright, even
   though upstream's own README example command uses exactly that trailing-slash form.
   Use `mistralai/Mistral-7B-Instruct-v0.2` (matches what oracle training itself writes
   into `args.yaml`, which is also what `get_target_lora_dirs` needs to match against).
8. **Two separate non-fatal crashes in upstream's own *automatic* post-training eval
   steps** — neither blocks anything we actually need, both happen only after the real
   artifact (adapter or `hypermod.pt`) is already safely saved to disk:
   - Oracle LoRA training's single-task self-eval (`sft_trainer.py`'s unconditional
     `eval_lora(..., full_eval=True)`) crashed for every one of our 8 tasks with a vLLM
     `ValueError: Unrecognized model in <run_dir>` — vLLM's LoRA tokenizer loader tries
     to read a full `config.json` from the adapter run directory, which only has
     `adapter_config.json` (PEFT format, no `model_type`). We looked for a way to
     disable this (`--eval_ds_info='{}'`) but that field also controls the train/val
     split used for early stopping, so clearing it would remove a real training-quality
     control — not worth it just to silence a cosmetic crash. Just expect and ignore
     this traceback; check for `adapter_model.safetensors` on disk instead of trusting
     the process's reported exit status.
   - Reconstruction training's automatic 10-benchmark eval (`recon_trainer.py`'s
     unconditional `eval_hypermod_checkpoint(..., full_eval=True)`) died silently with
     **no Python traceback at all** partway through generating eval LoRAs, right before
     it would have spun up a second full vLLM engine copy of Mistral-7B on top of
     whatever training had already allocated — consistent with an OOM kill, though we
     couldn't confirm via dmesg/journalctl (no kernel-log access on this shared
     machine). `hypermod.pt` was already written before this phase started, so it's a
     non-issue for us, but don't expect this automatic eval to complete on a
     single-GPU/shared-memory setup — plan to skip it or run our own smaller eval
     instead (which is exactly what our own harness's Step 4 does).
9. **Background job "failed"/exit-code-1 notifications on this machine are sometimes
   false.** Twice during Phase 2, a long-running job's wrapper reported failure
   (`/bin/bash: line 1: /tmp/claude-XXXX-cwd: Permission denied`) from a trailing
   housekeeping command unrelated to the actual job, while the real training/eval
   process was confirmed alive and progressing normally via `ps aux`/log files. Always
   verify via the actual process/output artifacts before trusting a reported failure on
   a long-running background command here.
10. For watching a long-running background job whose own wrapper's completion signal
    can't be trusted (#9), poll the real PID and output files directly with the
    `Monitor` tool (an `until`-style loop checking `kill -0 $PID` / grepping the log /
    checking for the output artifact) rather than relying on the task-notification
    event for that specific job.
11. **Dense-ΔW codecs can blow past 50GB for one training step at real model dimensions**
    if written the "natural" way — computing all `num_layers` layers via one batched
    `hypernetwork(condition_embeddings)` call and backpropagating a single combined loss
    keeps every layer's dense-ΔW graph (FourierFT's `ifft2` intermediates are ~1GB/layer
    at `q_proj`'s 4096x4096) alive simultaneously for that one backward pass. Profiled at
    56GB for a single step on the real 32-layer problem. `retain_graph=True` does **not**
    fix this — it disables buffer freeing for the *entire* graph reachable from each
    backward call, including that layer's own large buffers, not just a shared prefix.
    The actual fix: `TextToPeftHypernetwork.forward_layer` gives each layer an
    independent forward pass, so ordinary (non-retained) `backward()` frees that layer's
    graph before the next layer is even computed — bounds peak memory to ~one layer's
    worth regardless of `num_layers`. Separately, the *static* oracle ΔW targets
    themselves are large (7 tasks x 32 layers x 4096x4096 float32 for `q_proj` alone is
    ~15GB) — keep `oracle_targets` CPU-resident between folds and move only the current
    fold's target stack to the compute device once per fold (`pilot.py::_stack_targets`),
    not once per training step.
12. **`torch.fft.ifft2`'s default normalization (`norm="backward"`) divides by
    `out_features * in_features`** — negligible at the tiny dimensions
    `test_dynamic_t2p.py`'s unit tests use (8x8, N=64), but a ~1/16.7M attenuation at real
    Mistral-7B `q_proj` dimensions (4096x4096). This made `FourierFTCodec`'s
    `fourier_scaling` hyperparameter silently dimension-dependent — a `generated` value
    that produced a reasonable ΔW at toy scale required ~10,000x larger magnitude at real
    scale to produce the same ΔW, well past what a zero-initialized head can reach via
    gradient descent in any practical step budget (confirmed: 500 steps, lr up to 1e-2,
    train loss didn't move). Fixed by switching to `norm="forward"` (unnormalized ifft,
    all scaling folded into `self.scaling`), which makes `scaling` a true
    dimension-independent knob, matching every other codec's `alpha`-style scaling.
13. **After the `norm="forward"` fix, the codec became correspondingly *more* sensitive
    to head parameters, and the old default `learning_rate=1e-3` now diverges** (loss
    grew 10x within a handful of steps). Empirically, `lr=1e-6` is stable (an initial
    Adam-bias-correction overshoot at step 1, then a clean monotonic decrease) at real
    Mistral-7B dimensions; this is now `peft-hnet t2p-pilot`'s default. This is a genuine
    remaining rough edge, not a resolved one — see the Phase 3 section below for why 500
    steps at this conservative lr wasn't enough to clearly beat the zero-delta baseline,
    and what a follow-up should try (LR warmup, more steps, or better head init/scaling).
14. **Independent leave-one-out folds parallelize across GPUs via `--held-out-tasks`.**
    (Phase 3, retired code, kept for history.) `run_leave_one_out_pilot`'s folds don't
    share any state (each builds a fresh hypernetwork), so `peft-hnet t2p-pilot
    --held-out-tasks <subset>` let you launch one process per idle GPU, each computing a
    disjoint slice of the 8 folds against a different `CUDA_VISIBLE_DEVICES`/`--output`
    pair, then merge the `"folds"` lists from each output JSON. Cut the real Phase 3 run
    from ~73 min sequential to ~28 min across 3 GPUs. The general pattern (independent
    per-fold/per-task work parallelizes trivially across GPUs) still applies to Phase 4.
15. **LoRA and LoKr have a dead zero-gradient saddle point at this hypernetwork's default
    all-zero head init — found via the first real live-SFT smoke test, not a hypothetical.**
    Both split the head's flat `generated` output into two factors multiplied together
    (`ΔW = B @ A` for LoRA, a Kronecker product for LoKr) — bilinear in both factors.
    With `heads[name].weight` and `.bias` both zero-initialized (deliberate, so an
    untrained adapter contributes nothing), *both* factors are exactly zero, and the
    gradient w.r.t. *each* factor is proportional to the *other* — so both gradients
    vanish simultaneously too. Confirmed directly: `peft-hnet t2p-sft --representation
    lora` produced a loss curve that repeated *bit-for-bit* every 5 steps (one training
    epoch) — proof the parameters never moved at all, not just moved slowly. Standard
    LoRA/PEFT practice avoids this by initializing `A` non-zero and only `B` at zero;
    our hypernetwork's uniform zero-init broke that invariant. Fixed at the codec level,
    not the hypernetwork: `GeneratedUpdateCodec.initial_bias() -> Tensor | None` lets a
    codec override the head's bias (weight stays zero) — `LoRACodec`/`LoKrCodec` now
    return a bias with one factor's slice randomized and the other's left at zero
    (`ΔW` still exactly 0 at init, since the zero factor still zeroes the product, but
    gradient reaches the zero factor immediately). Every other codec (IA3, FourierFT,
    activation steering) is linear in `generated`, so all-zero is already fine for them
    and `initial_bias()` defaults to `None`. After the fix, the same smoke test's loss
    went from 14.0 to 1.1 over 60 real steps — a real, substantial decrease, not noise.
    This bug was latent through Phases 2/3 too (Phase 2's oracle LoRAs were trained via
    upstream's own separately-initialized `HyperModulator`, not ours; Phase 3's real
    pilot used FourierFT, which is unaffected) — this is the first time our own
    hypernetwork's LoRA path was actually gradient-trained end to end.
16. **Hooking a whole decoder layer (`"block"`) needs different output-unwrapping than
    hooking a linear submodule.** A `nn.Linear`'s forward hook receives/returns a bare
    tensor; a decoder layer's forward returns a tuple (`(hidden_states, ...)`).
    `TextToPeftHypernetwork.apply()`'s hook closure branches on
    `isinstance(output, tuple)` to unwrap/rewrap correctly either way — matches upstream
    Sakana's own `hooks.py::add_vec_hook` convention (`(newoutput, *output[1:])`), which
    is also where the `"block"` sentinel name itself comes from (not invented here).
17. **`tokenize_prompt_response`'s response-only label masking deliberately does not
    replicate upstream's `sequence_ids()`-based approach.** Upstream tokenizes prompt and
    response as a sequence *pair* (`tokenizer(prompt, response, ...)`) and uses
    `BatchEncoding.sequence_ids()` to mask sequence-0 (prompt) tokens — this depends on
    fast-tokenizer pair-tokenization semantics that aren't guaranteed the same way across
    every causal-LM tokenizer (that API is more commonly exercised for BERT-style
    sentence-pair tasks). Used a simpler, more portable equivalent instead: tokenize the
    prompt alone to get its token length, tokenize `prompt + response` as one string,
    mask the first that-many positions. Same supervision, no pair-tokenization edge cases
    to worry about across interpreter families.
18. **Qwen3's chat template defaults to "thinking" mode — `add_generation_prompt=True`
    alone is not enough to get a direct-answer prompt, and skipping this silently breaks
    both training and eval.** Found via the first multi-task pilot run: the
    `frozen_interpreter` baseline scored *below chance* on `boolq` (0.15 vs. 0.5 for a
    binary task). Root-caused by directly probing `_loglikelihood` on hand-written
    unambiguous cases (e.g. "Is Paris the capital of France?") — the model preferred
    "no" over "yes" by a nearly constant, content-independent margin regardless of the
    actual question. Cause: without `enable_thinking=False`, Qwen3's chat template
    leaves the prompt ending right after `<|im_start|>assistant\n`, with the model
    "expecting" to emit its own `<think>...</think>` block before answering — scoring (or
    training on) a direct answer glued on immediately after that point is badly
    out-of-distribution for the model's actual behavior at that point in the sequence.
    This affected *both* `t2p/lol_data.py::format_prompt_response` (training prompts) and
    `t2p/live_evaluator.py::_prompt` (eval prompts) — same bug, same fix, both call sites
    now pass `enable_thinking=False` explicitly, which is Qwen3's documented non-thinking
    mode (inserts a literal empty `<think>\n\n</think>\n\n` block into the prompt itself,
    a well-defined mode rather than an implicit one). Confirmed harmless as a no-op for
    non-Qwen3 tokenizers (Gemma-2, Mistral chat templates don't reference
    `enable_thinking` at all — extra `apply_chat_template` kwargs a template doesn't
    reference are silently ignored). After the fix, `frozen_interpreter` boolq jumped
    from 0.15 to a sane 0.75. This is a generally-relevant trap for *any* future work in
    this benchmark that scores a Qwen3 (or other hybrid-reasoning-model) interpreter via
    direct next-token/loglikelihood scoring rather than free-form generation — always
    check the frozen baseline against chance before trusting adapted-vs-frozen deltas.
19. **Moving this project's directory breaks every installed console script
    (`pytest`, `peft-hnet`, ...) — happened twice now (`scripts/peft_for_hnets` →
    `peft_for_hnets` → `AdapterBench`).** `uv`-installed entry-point scripts under
    `.venv/bin/` have an absolute-path shebang (`#!/old/path/.venv/bin/python`) baked in
    at install time; after a directory move that path no longer exists, so *any* command
    fails with a misleading `Failed to spawn: pytest — No such file or directory` (the
    error is about the shebang interpreter, not the script itself — `.venv/bin/pytest`
    still exists on disk). Symptom is easy to misdiagnose as a broken environment.
    **Fix: `uv pip install --python .venv/bin/python --reinstall -e ".[dev]"`** — this
    regenerates every entry-point shim (and any editable-install path metadata, e.g. an
    `Unnecessary package: peft-hnet-benchmark==0.1.0 (from file:///old/path)` line in
    `uv run -v`'s debug output is the tell) against the new path, without needing to
    recreate the whole venv from scratch. Also note: **editing `pyproject.toml`'s
    `[project].name`** can independently trigger `uv run` to silently re-resolve
    `uv.lock` against whatever Python `uv` finds first on `PATH`/`VIRTUAL_ENV` (on this
    machine, an unrelated conda env's Python 3.14, incompatible with the pinned
    `torch==2.5.1` wheels) — if a rename is immediately followed by
    `torch ... doesn't have a source distribution or wheel for the current platform`,
    re-lock explicitly against the project's own interpreter:
    `uv lock --python .venv/bin/python`.

## Roadmap (not yet done, in priority order)

### Phase 2: train our own hypernetwork checkpoint — ✅ done (see Status snapshot above)

### Phase 3: generalize the representation seam — ✅ done, then retired and archived (see Phase 4)

**Superseded.** Kept below for history — the reconstruction-matching machinery this
phase built (`t2p/pilot.py`, `t2p/oracle_targets.py`) has been removed from the active
codebase (recoverable via git history, not a live `archive/` directory). Live end-to-end
SFT (Phase 4) is now the sole training/eval
mechanism, for a reason that goes beyond "it's what Sakana actually does": reconstruction
matching requires a representation-specific *oracle target* (an already-trained example
of that representation to regress onto), which has no general recipe for activation-based
representations the way "fine-tune a LoRA" does for weight-based ones. Live SFT needs no
oracle at all — see Phase 4 below.

The point of this phase is **not** picking a winning PEFT representation — it's proving
the hypernetwork/codec seam is genuinely representation-agnostic: any codec with a
weight-space update can be dropped into the same reconstruction training/eval loop with
no new plumbing per representation. That seam is now implemented and exercised
end-to-end against real Mistral-7B-dimension oracle LoRAs (not just toy `nn.Linear`
layers), closing what the Status snapshot called "the biggest remaining gap."

Key simplifying fact that made this cheap to build: **the core test needs no live 7B
model, no forward hooks, and no new oracle training** — it's pure parameter-space
regression, reusing the exact 8 oracle LoRAs Phase 2 already trained. Every task's
reconstruction target is just `ΔW = B @ A` (dense, representation-agnostic), computable
directly from the existing `adapter_model.safetensors` files.

All four originally-scoped gaps are closed:
1. **Dense ΔW exposure** — every codec except IA3 now has `dense_delta(generated,
   layer_index) -> (batch, out, in)` (`t2p/codecs.py`).
2. **IA3 exclusion** — confirmed as a scoping decision, not a bug: `IA3Codec.dense_delta`
   raises `NotImplementedError` with an explanation (multiplicative on activations, no
   weight-space delta).
3. **Oracle ΔW target loading** — `t2p/oracle_targets.py` ports upstream's
   `get_recon_train_data` key-parsing + `bmm(B,A)` logic; `find_oracle_adapter_paths`
   resolves task ids to their timestamped oracle-LoRA run directories.
4. **Task-description embedding** — `t2p/condition_encoder.py` embeds via
   `Alibaba-NLP/gte-large-en-v1.5` **in-process** (confirmed compatible with `peft_hnet`'s
   own pinned `transformers==4.57.6`, given `trust_remote_code=True` — no cross-venv
   bridge needed, simpler than `ReleasedTextToLoRABackend`'s pattern).

Also confirmed: `src/peft_hnet/generators.py`/`reconstruction.py` (an earlier
condition→flat-vector reconstruction training loop) really is dead weight here —
`TextToPeftHypernetwork.forward` returns a `dict[str, Tensor]` keyed by module name with
a different shape per module, not the flat `(batch, state_dim)` shape that loop expects.
Left in place but not used by the Phase 3 pilot; candidate for deletion if nothing else
claims it.

**Real pilot run**: `TextToPeftHypernetwork` configured for `FourierFTCodec` (the paper's
flagship non-LoRA case), leave-one-task-out across all 8 real oracle LoRAs (train on 7,
evaluate held-out reconstruction L1 against a predict-zero-delta baseline), 500 steps,
lr=1e-6, `delta_w_scaling=10000` (matches Phase 2's `train_hyper_recon.py` convention).
Ran as 3 parallel processes across 3 idle GPUs via `--held-out-tasks` (see gotcha #14),
merged into `results/t2p_pilot/fourierft_vs_lora_delta.json`:

| held-out task | hypernetwork L1 | zero-baseline L1 | beats zero |
|---|---|---|---|
| lol_022 | 0.1779 | 0.1730 | no (+2.80%) |
| lol_033 | 0.3313 | 0.3285 | no (+0.87%) |
| lol_034 | 0.3655 | 0.3625 | no (+0.82%) |
| lol_035 | 0.2346 | 0.2297 | no (+2.11%) |
| lol_039 | 0.2551 | 0.2512 | no (+1.54%) |
| lol_043 | 0.4659 | 0.4633 | no (+0.55%) |
| lol_044 | 0.5527 | 0.5502 | no (+0.45%) |
| lol_045 | 0.4833 | 0.4816 | no (+0.36%) |

**Honest read: 0/8 folds beat the zero-delta baseline, and this is a hyperparameter
problem, not a representational one.** Two real bugs were found and fixed while getting
here (gotchas #11-12: a memory blowup from the natural way to backprop through all
layers, and a genuine `torch.fft.ifft2` normalization bug that made `FourierFTCodec`
practically untrainable at real dimensions). Fixing the second bug fixed the *gradient
magnitude* problem but also made the codec much more sensitive to its head parameters,
which made the old default learning rate diverge — the safe lr found empirically
(1e-6, gotcha #13) is conservative enough that 500 steps isn't enough to move the
hypernetwork meaningfully away from its zero-initialized starting point (train loss on
the 7 training tasks barely moves in any fold). This was confirmed on a synthetic
sanity check too (shared-target small-dimension case in `tests/test_pilot.py` *does*
clearly beat the zero baseline, proving the training machinery itself works when given
an easy, learnable signal and a normal lr) — so the gap here is specifically "the real
model dimensions force a lr too conservative for 500 steps," not a bug in the loop.
Follow-up options, in likely order of effort: (a) just run far more steps at the same
conservative lr, (b) an lr warmup schedule to avoid wasting the first several steps on
Adam's bias-correction overshoot, (c) a smarter head initialization tied to the codec's
actual output scale instead of the current uniform zero-init. None of these require
further architecture changes — the harness is done; this is a training-recipe tuning
pass whenever the project returns to Phase 3.

### Phase 4: live end-to-end SFT — ✅ done (mechanism, real smoke test, and the small multi-task pilot with real downstream eval)

Restructured the benchmark around Sakana's actual main training method (live-hook SFT,
not reconstruction matching) specifically to make adding a new representation
"almost plug-and-play": a researcher needs only an output structure (a codec) and a hook
site (a linear submodule name, or `"block"` for the whole decoder layer), and the same
training loop, data pipeline, and evaluator work unchanged. Full architecture in the
section above; this section tracks what's been validated vs. what's still ahead.

**Done:**
1. **Mechanism generalization** (`t2p/codecs.py`, `t2p/hypernetwork.py`) — hook sites
   generalized beyond named linear submodules to whole decoder layers, proven
   differentiably with a synthetic tuple-returning fake layer before touching any real
   model (`tests/test_dynamic_t2p.py`).
2. **Real single-task smoke test** — `peft-hnet t2p-sft` against `Qwen/Qwen3-0.6B` (real
   model, real forward/backward passes) and one real `lol_022` task from the actual
   Lots-of-LoRAs/SNI data:
   - LoRA (`--target-modules q_proj,v_proj`): loss 14.02 → 1.11 over 60 real steps.
   - Activation steering (`--target-modules block`, the new `ActivationSteeringCodec`):
     loss 15.69 → 1.31 over 60 real steps — **the same command, same training loop, only
     `--representation`/`--target-modules` changed.** This is the actual proof of the
     plug-and-play claim, not just an architectural argument for it.
   Found and fixed one real, previously-latent bug along the way (gotcha #15 — LoRA/LoKr's
   bilinear zero-gradient saddle point).

3. **Small multi-task pilot with real downstream evaluation** — `peft-hnet t2p-sft-pilot`
   (new CLI command): trains LoRA, IA3, and `ActivationSteeringCodec` independently, each
   to 400 real live-SFT steps on the shared 8-task training split
   (`lol_022, lol_033, lol_034, lol_035, lol_039, lol_043, lol_044, lol_045`), then scores
   each via `HypernetworkDownstreamEvaluator` against 20 real held-out examples per family
   from `boolq`/`hellaswag` (`task_examples.py`), alongside a `frozen_interpreter`
   baseline. First genuinely honest "does representation X actually help downstream,
   trained live, with no oracle" result:

   | representation | generated params | train loss (init → final) | boolq (n=20) | hellaswag (n=20) |
   |---|---|---|---|---|
   | frozen_interpreter | 0 | — | 0.75 | 0.20 |
   | lora (`q_proj,v_proj`) | 1,146,880 | 5.92 → 2.85 | 0.15 | 0.10 |
   | ia3 (`k_proj,v_proj,down_proj`) | 86,016 | 5.92 → 0.46 | 0.85 | 0.25 |
   | activation_steering (`block`) | 28,672 | 5.92 → 3.85 | 0.85 | 0.25 |

   **Honest read, small n caveats aside:** LoRA reaches a middling training loss but
   *actively destroys* downstream performance on both held-out families — well below the
   frozen baseline and, on `boolq`, below random guessing (0.15 vs. 0.5 chance for a
   binary task). This echoes the Phase 1 "notable finding" above (an adapter can suppress
   a capability the base model already had, worse outside its conditioning distribution)
   but is more severe here: both eval families' condition embeddings are genuinely
   out-of-distribution relative to the 8 `lol_*` training descriptions, and LoRA's
   dense, 40x-larger-than-steering parameterization apparently overfits hard enough to
   that narrow conditioning distribution to actively harm generalization. IA3 and
   activation steering — the two multiplicative/additive, much-lower-parameter-count
   representations — both *improve* over frozen on both families, and reach a
   dramatically lower training loss (IA3 especially, 0.46) with 13-40x fewer generated
   parameters than LoRA. IA3 and activation steering land on identical accuracy figures
   (0.85/0.25) — plausibly real convergent behavior given both are far cheaper/simpler
   representations than LoRA here, but with n=20 per family this could also just be
   coincidence at this sample size; don't read the tie as more than "both clearly beat
   LoRA and frozen here." Not a competitive benchmark result (Qwen3-0.6B, 8 tasks, 400
   steps, n=20/family) — the point was proving the plug-and-play train+eval loop produces
   real, differentiated, non-degenerate signal across three structurally different
   representations without any per-representation plumbing beyond the codec + hook site.
   Follow-up worth doing before trusting these numbers further: larger n, more seeds,
   and specifically investigating *why* LoRA generalizes so much worse here (candidates:
   its much larger parameter count overfitting the 8-task conditioning distribution
   harder than IA3/steering; the bilinear zero-gradient fix (gotcha #15) still leaving
   LoRA's optimization landscape rougher than the other codecs'; or an interaction with
   the `enable_thinking` fix below that happens to hurt LoRA specifically — not yet
   isolated).

**Explicitly deferred, not part of this pass:**
- The full 479-task decontaminated training config (`configs/hyper_lora_decontam_lol_tasks.yaml`)
  and distributed/`torchrun` training (`text_to_peft_gemma2b_sft.yaml`'s original scope,
  `runtime.launcher: torchrun, num_processes: 8`) — the pilot above deliberately stays
  small-scale first, matching every prior phase's pattern.
- Dense-ΔW representations (FourierFT, LoKr) at live-SFT scale on a real multi-billion
  parameter interpreter — materializing a dense ΔW inside a live forward graph is a real
  memory risk structurally different from (worse than) what Phase 3 already fixed for
  the retired reconstruction path: there, `forward_layer`'s per-layer-then-backward trick
  worked because the loss was computed independently per layer; in live SFT, one shared
  next-token loss only exists after every layer's hook has run, so every earlier layer's
  dense-ΔW graph must stay alive simultaneously regardless. Needs its own memory
  validation pass before scaling past cheap codecs (LoRA, IA3, activation steering).
- Reproducing the paper's claimed GPT-4o-mini-generated task descriptions — unresolved,
  see `t2p/lol_data.py`'s module docstring; using the existing stored descriptions as-is
  for now per project decision.
- Unifying `HFDownstreamEvaluator`/`HypernetworkDownstreamEvaluator` into one class (they
  currently duplicate the scoring routines) — kept separate deliberately for now so the
  new live-hook path can't destabilize the already-working PEFT-artifact baseline path.

### Phase 5: lightweight synthetic setting — ✅ done

A second, lightweight setting for the live-SFT mechanism (`peft-hnet t2p-synthetic-pilot`),
purely additive to the real Qwen3/Lots-of-LoRAs setting above. Motivation: that real
setting is heavy (network access, a real ~0.6B-parameter model) and has no ground truth —
when a representation underperforms (LoRA, in the Phase 4 pilot) there's no way to tell
whether the *codec* is underpowered or the *real-world task* is just hard. This setting
reuses the entire training/codec stack unchanged and swaps only the interpreter and data
source for ones with exactly known correct answers, closing that gap.

- **Tiny real-`transformers` interpreter** (`t2p/tiny_interpreter.py::build_tiny_interpreter`)
  — a `LlamaConfig`/`LlamaForCausalLM` at tiny dimensions (2 layers, hidden_size 32),
  constructed via `from_config` and **never downloaded** — fully network-independent,
  randomly initialized, real `transformers` mechanics (`.generate()`, forward hooks,
  `get_decoder_layers`/`infer_module_shapes` all work completely unchanged). Llama was
  chosen over reusing Qwen3 specifically to avoid the `enable_thinking` chat-template trap
  (gotcha #18) — moot here anyway since this setting never uses a chat template at all;
  examples are built directly as integer ids over a fixed 16-symbol vocabulary
  (`PAD/BOS/EOS/SEP` + digits 0-9), no tokenizer involved on the interpreter side.
- **Synthetic task registry** (`t2p/synthetic_tasks.py::TASK_FAMILIES`) — 5 hard-coded task
  families, each a pure-Python transform with exactly known correct output plus 4
  hand-written description paraphrases (no external data, no description-generation
  provenance gap to track): `copy` (identity), `reverse`, `increment` (+1 mod 10 per
  digit, pointwise), `sort` (ascending, needs real cross-position reasoning), `constant`
  (fixed output regardless of input — a control task testing whether conditioning can
  make the model ignore its input at all). `SyntheticSFTDataset` mirrors `LolSFTDataset`'s
  random(train)/deterministic(eval) description-embedding draw convention exactly, but
  generates examples on the fly via a per-`(seed, family, index)` RNG — no dataset
  download, fully reproducible. Batches collate via the *existing*
  `t2p/lol_data.py::lol_collate_fn` **completely unchanged** — good evidence the pipeline
  was already factored data-source-agnostically despite that function's name. Condition
  embeddings reuse `t2p/condition_encoder.py` unchanged too (real `gte-large-en-v1.5`) —
  the *only* things that differ from the real setting are the interpreter and the data.
- **Synthetic evaluator** (`t2p/synthetic_evaluator.py`) — function-based (not a class;
  proportionate to how little logic remains with no chat template or text parsing):
  greedy-decode via `interpreter.generate()` under the same `hypernetwork.apply(...)`
  contextmanager every setting uses, then direct token-id exact-match against the known
  target. A `condition_embeddings=None` path scores the frozen baseline.
- **CLI**: `peft-hnet t2p-synthetic-pilot` mirrors `t2p-sft-pilot`'s structure (shared
  interpreter/encoder/batches, per-representation train-then-evaluate loop, incremental
  `results.jsonl`/`loss_curves.json` writes) but defaults `--device cpu` (no GPU needed at
  all) and — affordable for the first time at this tiny scale — defaults
  `--representations` to **all six** registered codecs, including FourierFT/LoKr, which
  Phase 4 explicitly deferred from live SFT at real-model scale over a memory risk that is
  specific to real model dimensions and remains open there.
- **Schema**: `schema.py::DatasetSpec.source` gained `"synthetic"` (additive); new
  `configs/setups/synthetic_sft_pilot.yaml`.
- **A real bug found immediately by the first real run**: `FourierFTCodec`'s default
  `n_frequency=1000` exceeds this tiny model's smallest target module's element count
  (`v_proj` is 32×16=512) — `make_codec` raised `n_frequency exceeds the matrix size`.
  Fixed by exposing `--n-frequency`/`--rank` on the CLI with tiny-scale-safe defaults
  (`n_frequency=32`, `rank=4`) rather than hardcoding the real-model defaults (1000/8).

**Real results** (200 examples/family training, 1000 total, batch size 16; `n=50`/family
eval; two runs, 400 and 1500 steps, to see whether more budget changes the picture):

| representation | copy | reverse | increment | sort | constant |
|---|---|---|---|---|---|
| frozen_interpreter | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| lora (400 / 1500 steps) | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.00 / 1.00 |
| freeze_a_lora | 0.00 / 0.02 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.00 / 1.00 |
| ia3 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.00 / 1.00 |
| lokr | 0.04 / **0.60** | 0.00 / 0.00 | 0.00 / **0.66** | 0.00 / 0.00 | 1.00 / 1.00 |
| fourierft | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.28 / 0.82 |
| activation_steering | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.00 / 1.00 |

**Honest read:** `constant` (the control task — ignore the input entirely) is solved
almost immediately by every representation except FourierFT, confirming the conditioning
mechanism itself works — the hypernetwork can clearly steer the tiny interpreter's
behavior via the codec/hook path. The genuinely interesting result is what happens on the
*input-dependent* tasks: at 1500 steps, **LoKr is the only representation that learns
`copy` and `increment`** (both pointwise, position-preserving transforms) — a real,
substantial jump (0.60/0.66) that LoRA, FreezeALoRA, IA3, and activation steering never
make at the same budget, despite LoKr sharing the same bilinear-factorization fix (gotcha
#15) as LoRA. Nothing solves `reverse`/`sort` (both require reordering across positions,
plausibly beyond a 2-layer/hidden-32 model's capacity regardless of representation, or
just needing far more steps). FourierFT remains the weakest representation even on the
control task (0.28 → 0.82, still not fully solved) — consistent with the historical
gotchas #12/13 about FourierFT being unusually sensitive to head-parameter scaling; worth
retrying here with a scaling/lr sweep before concluding it's a capacity problem rather
than the same optimization-recipe sensitivity found before. This is exactly the kind of
finding this setting was built to produce: a representation-specific effect, isolated
from real-world task noise, that the real Qwen3 setting's small held-out benchmark
couldn't have distinguished from "the benchmark task was just hard."

**Explicitly out of scope for this pass** (per the approved plan): leave-one-family-out
generalization testing (cheap to add later given synthetic data, not built now); no
changes to the real Qwen3/Lots-of-LoRAs setting.

### Phase 6: Doc-to-LoRA as the second benchmark "setting"

`doc-to-lora/` is cloned but has zero integration so far (no config, no backend, no code
references anywhere). Structurally different from T2L (Perceiver-style cross-attention
over document token activations, rather than a pooled task-description embedding) — will
likely need its own `HypernetworkBackend` implementation following the same
subprocess-bridge pattern as `ReleasedTextToLoRABackend`, not a reused one.

### Smaller/deferred items

- `interpreter_with_icl` and `oracle_adapter` baselines (`BENCHMARK_CONTRACT.md` lists
  these; Phase 1 only has `lora` + `frozen_interpreter`).
- Description-variant robustness: `args.yaml::eval_ds_info` has 3 paraphrased
  descriptions per task; Phase 1 only used variant 0.
- `README.md` still documents a `peft-hnet synthetic --config configs/standard.json`
  command that no longer exists (leftover from the removed synthetic pilot) — fix next
  time README.md is touched.

## Reference: prior art this benchmark builds on

- Program-as-Weights: arXiv:2607.02512
- Text-to-LoRA: arXiv:2506.06105
- Doc-to-LoRA: arXiv:2602.15902 — github.com/SakanaAI/doc-to-lora
- FourierFT: arXiv:2405.03003 · KronA: arXiv:2212.10650 · Compacter: arXiv:2106.04647
- HyperTuning/HyperLlama (Phang et al.): arXiv:2402.16817 — the one directly-comparable
  prior result that *disagrees* with T2L/PAW's LoRA-over-steering-tokens finding
  (opposite conclusion, different task distribution). Characterizing when each wins is
  itself a legitimate research question this benchmark can eventually answer.
