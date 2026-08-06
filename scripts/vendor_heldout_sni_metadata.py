"""Vendor metadata.yaml for the held-out SNI eval tasks that ship without it.

Of T2A's 21 held-out `lol_###` validation tasks (`eval_ds_info` in the decontam yaml), only 10 have a
vendored `data/t2a/tasks/<id>/metadata.yaml`. The other 11 have their 3 held-out descriptions in
`eval_ds_info` but no dataset kwargs. This resolves each to its `Lots-of-LoRAs/task###_*` Hub dataset,
verifies it loads + preprocesses in the SNI convention, and writes a metadata.yaml matching the shipped
schema so `t2a_eval_heldout_sni.py` can score the full 21. Descriptions come from `eval_ds_info` (the
canonical held-out eval descriptions); template/response_field match the vendored tasks.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml
from datasets import load_dataset
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adapterbench.cli._shared import T2A_DECONTAM_CONFIG  # noqa: E402
from adapterbench.t2a.lol_data import preprocess_lol_example  # noqa: E402

TASKS_ROOT = Path("data/t2a/tasks")


def main() -> None:
    decontam = yaml.safe_load(Path(T2A_DECONTAM_CONFIG).read_text())
    eval_ds_info = decontam["eval_ds_info"]
    lol = [k for k in eval_ds_info if str(k).startswith("lol_")]
    missing = [t for t in lol if not (TASKS_ROOT / t / "metadata.yaml").exists()]
    print(f"held-out lol_ tasks: {len(lol)}; missing metadata: {len(missing)} -> {missing}")

    print("listing Lots-of-LoRAs datasets on the Hub...", flush=True)
    names = {d.id.split("/")[-1] for d in HfApi().list_datasets(author="Lots-of-LoRAs", limit=5000)}

    written, failed = [], []
    for task_id in missing:
        num = task_id.split("_")[1]  # 'lol_701' -> '701'
        hits = sorted(x for x in names if re.match(rf"task0*{num}_", x))
        if not hits:
            print(f"  [FAIL] {task_id}: no task{num}_* on the Hub"); failed.append(task_id); continue
        full = hits[0]
        path = f"Lots-of-LoRAs/{full}"
        try:
            raw = load_dataset(path=path, split="train", name="default")
            _ = preprocess_lol_example(raw[0])  # verify the SNI parse works on this task
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {task_id} ({path}): {type(e).__name__} {str(e)[:120]}"); failed.append(task_id); continue

        meta = {
            "assistant_prefill": "",
            "descriptions": eval_ds_info[task_id]["descriptions"],
            "ds_kwargs": {"name": "default", "path": path, "split": "train[:10000]"},
            "response_field": "answer",
            "system_message": "",
            "task_name": full,
            "user_prompt_template": "{task_def}\n\n{problem}",
        }
        out = TASKS_ROOT / task_id / "metadata.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(meta, sort_keys=True, allow_unicode=True))
        print(f"  [ok] {task_id:10} -> {full}  (n={len(raw)})", flush=True)
        written.append(task_id)

    print(f"\nwrote {len(written)} metadata.yaml; failed {len(failed)}: {failed}")


if __name__ == "__main__":
    main()
