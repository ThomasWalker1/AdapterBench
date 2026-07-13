# Vendored Text-to-LoRA data

These files are vendored (sliced) from the **Text-to-LoRA** repository so the AdapterBench
T2L setting is self-contained and needs no `upstream/` clone at run time:

- Source: `https://github.com/SakanaAI/text-to-lora`
- Commit: `8ba77493eb8c732bac41c22ca76117dcacfcf2a4`
- License: **Apache-2.0** (redistributed here under that license; see the upstream repo for the
  full license text and attribution).

## Contents

- `hyper_lora_decontam_lol_tasks.yaml` — Text-to-LoRA's decontaminated task split
  (`train_ds_names`: the 479 Lots-of-LoRAs training tasks, plus its held-out validation and
  contamination-removed lists). Read by `t2p/lol_data.py::load_decontaminated_train_task_ids`
  / `validate_training_tasks`.
- `tasks/<task_id>/metadata.yaml` — per-task descriptions + examples for the Lots-of-LoRAs
  tasks (511 tasks). Read by `t2p/lol_data.py::load_task_metadata` and `LolSFTDataset`. The
  Lots-of-LoRAs tasks themselves derive from Super-Natural-Instructions.
- `eval_ds_info.yaml` — just the `eval_ds_info` block sliced from
  `trained_t2l/gemma_2b_t2l/args.yaml` (held-out benchmark task *descriptions* only; the eval
  *examples* are built at run time from public HF datasets by `task_examples.py`). Read by
  `task_examples.py::load_task_descriptions`.

Every `t2p-*` CLI command defaults to these paths (`cli/_shared.py::T2L_*`). To refresh from a
different upstream commit, re-slice `tasks/` + `configs/` + `eval_ds_info` and overwrite here.

## Trimming (not a verbatim slice)

To keep the vendored copy lean, this directory is reduced from the upstream `tasks/` in two
ways; both are safe for the AdapterBench T2L recipe and reversible by re-slicing from the
pinned commit:

- **Only the 479 decontaminated *training* tasks are kept.** Upstream ships 511 task dirs; the
  extra 32 (the benchmark-eval tasks — arc/boolq/hellaswag/gsm8k/humaneval/mbpp/… — plus
  held-out-validation and contamination-removed `lol_*`) are never read: training loads only
  `train_ds_names`, and evaluation sources its task descriptions from `eval_ds_info.yaml` and
  its examples from Hugging Face.
- **Each task's `descriptions` list is truncated to the first 128** (upstream ships ~200). The
  T2L recipe samples at most 128 descriptions per task (`--max-descriptions 128`), so the
  remainder is never used.

Training *examples* are not vendored: each `metadata.yaml`'s `ds_kwargs.path` is a Hugging Face
Hub id (e.g. `Lots-of-LoRAs/task022_...`), fetched at run time — so no `upstream/` clone is
needed, only network access to the Hub (the same dependency the eval already has).
