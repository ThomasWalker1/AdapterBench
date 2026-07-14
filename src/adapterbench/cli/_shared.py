"""Shared plumbing for the CLI command modules.

Every command that loads a frozen interpreter, embeds task-description conditions,
or streams results to disk went through a byte-for-byte copy of the same block
before this module existed; extracting them here keeps that plumbing in one place
without changing any command's behavior. Command *logic* stays in the per-command
modules (`meta.py`, `reproduction.py`, `live_sft.py`); only the mechanical setup
lives here.
"""

from __future__ import annotations

import json
from pathlib import Path

# repo root is four parents up from src/adapterbench/cli/_shared.py
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = REPO_ROOT / "configs"

# Vendored Text-to-LoRA data (see data/t2l/NOTICE.md) - the Lots-of-LoRAs per-task
# metadata, the decontaminated 479-task train split, and the held-out eval-task
# descriptions, sliced from SakanaAI/text-to-lora at the pinned commit. Vendored into the
# repo so the T2L setting is self-contained and needs no `upstream/` clone (which is
# gitignored and gets cleaned). Every t2p-* command defaults to these paths.
T2L_DATA_DIR = REPO_ROOT / "data" / "t2l"
T2L_TASKS_DIR = T2L_DATA_DIR / "tasks"
T2L_DECONTAM_CONFIG = T2L_DATA_DIR / "hyper_lora_decontam_lol_tasks.yaml"
T2L_EVAL_DESCRIPTIONS = T2L_DATA_DIR / "eval_ds_info.yaml"

# lol_022/043/044/045/047/050/063/064 are all confirmed present in T2L's own
# train_ds_names (data/t2l/hyper_lora_decontam_lol_tasks.yaml).
# The previous default (lol_022,033,034,035,039,043,044,045) trained on lol_033/034
# (two of T2L's 10 contamination-removed tasks) and lol_035/039 (two of T2L's own 11
# held-out zero-shot validation tasks) - exactly the leakage a training pilot should
# avoid. --decontam-config validates any --tasks value against this list at run time.
DEFAULT_SFT_TRAIN_TASKS = "lol_022,lol_043,lol_044,lol_045,lol_047,lol_050,lol_063,lol_064"

# Default hook site per adapter, shared by every live-SFT pilot/sweep command so they
# never drift apart. LoRA (the only baseline codec) modifies attention projections; as
# new codecs are added one at a time as reviewed codecs; they register their own hook site
# here (e.g. activation steering -> ["block"], IA3 -> ["k_proj", "v_proj", "down_proj"]).
PILOT_DEFAULT_TARGET_MODULES = {
    "lora": ["q_proj", "v_proj"],
}

# TextToPeftHypernetwork's own default (never overridden by any existing t2p-sft*
# command either) - kept as a plain module constant rather than a new CLI flag so
# DocumentPerceiverConditioner's task_dim (= latent_dim // 2, see hypernetwork.py's
# comment on that requirement) stays in lockstep with the hypernetwork's own trunk
# width without the two ever being passed independently.
D2P_LATENT_DIM = 512


def load_frozen_interpreter(interpreter_id: str, device: str):
    """Load the interpreter in bf16 on `device`, put it in eval mode, freeze every
    parameter, and return `(tokenizer, interpreter, decoder_layers)`.

    This is the exact setup every live-SFT command shares. Note it deliberately does
    NOT `.resolve()` the interpreter id (it's a HF model name, not a path). `eval()`
    here freezes the *interpreter's* dropout only - each command still calls
    `hypernetwork.eval()` separately before held-out scoring (PROJECT_PLAN gotcha #14).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..t2p.model_utils import get_decoder_layers

    tokenizer = AutoTokenizer.from_pretrained(interpreter_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    interpreter = AutoModelForCausalLM.from_pretrained(interpreter_id, dtype=torch.bfloat16).to(device)
    interpreter.eval()
    for parameter in interpreter.parameters():
        parameter.requires_grad = False
    layers = get_decoder_layers(interpreter)
    return tokenizer, interpreter, layers


def embed_training_conditions(condition_encoder: str, device: str, tasks_dir: Path, task_ids, max_descriptions: int):
    """Load the condition encoder and embed each training task's descriptions.

    Returns `(encoder_model, encoder_tokenizer, metadata_by_task, embeddings_by_task,
    condition_dim)`. The encoder model is returned rather than deleted so a caller that
    also needs to embed held-out eval descriptions (t2p-sft-pilot, t2p-sft-sweep) can
    reuse it; every caller is responsible for `del encoder_model` once done, exactly as
    the inline code did.
    """
    from ..t2p.condition_encoder import embed_task_descriptions, load_condition_encoder
    from ..t2p.lol_data import load_task_metadata

    encoder_model, encoder_tokenizer = load_condition_encoder(condition_encoder, device=device)
    metadata_by_task = {
        task_id: load_task_metadata(tasks_dir, task_id, max_descriptions=max_descriptions)
        for task_id in task_ids
    }
    embeddings_by_task = {
        task_id: embed_task_descriptions(metadata.descriptions, encoder_model, encoder_tokenizer).to(device)
        for task_id, metadata in metadata_by_task.items()
    }
    condition_dim = next(iter(embeddings_by_task.values())).shape[-1]
    return encoder_model, encoder_tokenizer, metadata_by_task, embeddings_by_task, condition_dim


class ResultRecorder:
    """Accumulate `EvaluationResult`s, re-write results.jsonl/.csv after each one, and
    print a per-result progress line.

    Every command streamed results to disk this way (write-after-each so a killed run
    still leaves partial, valid output); only the printed line's field layout differed,
    so that is the one thing a caller passes in via `format_line`.
    """

    def __init__(self, output_dir: str | Path, format_line):
        from ..reporting import write_results

        self._write_results = write_results
        self.output_dir = Path(output_dir)
        self.results: list = []
        self.format_line = format_line

    def record(self, result) -> None:
        self.results.append(result)
        self._write_results(self.results, self.output_dir)
        print(self.format_line(result), flush=True)


def write_json(path: str | Path, payload) -> None:
    """Write `payload` as pretty JSON (with trailing newline), creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
