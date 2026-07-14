import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "d2p_niah_aggregate.py"
SPEC = importlib.util.spec_from_file_location("d2p_niah_aggregate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_run(root: Path, seed: int, matched: float, control: float, transition: int) -> None:
    run = root / f"s{seed}"
    run.mkdir(parents=True)
    result = {
        "adapter": "lora",
        "task_id": "niah_384",
        "metrics": {"accuracy": matched, "accuracy_ctxswap": control},
    }
    (run / "results.jsonl").write_text(json.dumps(result) + "\n")
    history = {
        "lora": [
            {"step": transition - 500, "niah_384": {"accuracy": 0.25}},
            {"step": transition, "niah_384": {"accuracy": 0.5}},
        ]
    }
    (run / "history.json").write_text(json.dumps(history))


def test_aggregate_reads_headline_and_transition_per_seed(tmp_path):
    _write_run(tmp_path, 777, matched=1.0, control=0.0, transition=2000)
    _write_run(tmp_path, 778, matched=0.75, control=0.125, transition=2500)

    grouped = MODULE.aggregate(tmp_path)

    results = grouped[("lora", "niah_384")]
    assert [result.run for result in results] == ["s777", "s778"]
    assert [result.headline for result in results] == [1.0, 0.625]
    assert [result.transition_step for result in results] == [2000, 2500]
