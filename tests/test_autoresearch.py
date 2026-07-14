"""Unit tests for the per-codec autoresearch driver (CPU, no GPU).

Covers the guarantees the leaderboard depends on: the HP partition is enforced (only FREE HPs are
searchable — a codec can't cheat by tuning the substrate), best-of selection maximizes the
matched−control objective (never loss), aggregation is multi-seed, the search budget/space is
reported, and launch mode is restart-safe (skips existing cells). The I2P objective extractor is
checked against the real record shape; the end-to-end selection on real data is validated by
`scripts/autoresearch.py` against results/i2p_hypernoise_v2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adapterbench.autoresearch import (
    FREE, SHAPE_IDENTITY, SUBSTRATE, SettingSpec, launch_missing_cells, search_over_cells,
)
from adapterbench.autoresearch_settings import I2P_SPEC


def _i2p_cell(scale, reg, seed, ir_gain, reward_target="imagereward"):
    return {"metadata": {"reward_target": reward_target, "lora_scale": str(scale),
                         "reg_weight": reg, "seed": seed, "rank": 16, "steps": 3000},
            "metrics": {"reward_imagereward_gain": ir_gain}}


# ── partition enforcement ────────────────────────────────────────────────────────────────────
def test_search_space_rejects_non_free_hps():
    with pytest.raises(ValueError, match="not FREE"):
        I2P_SPEC.validate_search_space({"rank": [8, 16]}, {})  # shape-identity — must not be searched
    with pytest.raises(ValueError, match="not FREE"):
        I2P_SPEC.validate_search_space({"scale": [2, 4]}, {"batch_size": 4})  # substrate pinned by loop
    with pytest.raises(ValueError, match="not FREE"):
        I2P_SPEC.validate_search_space({"nonexistent_hp": [1]}, {})
    # a purely-FREE search space is accepted
    I2P_SPEC.validate_search_space({"scale": [2, 4]}, {"reg_weight": 0.25})


def test_partition_classes_are_exhaustive_and_disjoint():
    for spec in (I2P_SPEC,):
        classes = set(spec.hp_classes.values())
        assert classes <= {FREE, SUBSTRATE, SHAPE_IDENTITY}
        assert spec.hps_of_class(FREE) and spec.hps_of_class(SHAPE_IDENTITY)


# ── best-of selection on matched−control, multi-seed ─────────────────────────────────────────
def test_best_of_selects_max_objective_over_seeds():
    # scale 4 is the true optimum; each scale has 2 seeds. Also throw in a red (control) cell at
    # scale 4 that must be ignored by the objective (reward_target != imagereward).
    cells = [
        _i2p_cell(2, 0.25, 777, 0.10), _i2p_cell(2, 0.25, 778, 0.12),
        _i2p_cell(4, 0.25, 777, 0.18), _i2p_cell(4, 0.25, 778, 0.16),
        _i2p_cell(16, 0.25, 777, -1.20), _i2p_cell(16, 0.25, 778, -1.10),
        _i2p_cell(4, 0.25, 777, -3.30, reward_target="red"),  # control — must not enter the objective
    ]
    report = search_over_cells(I2P_SPEC, cells, search_dims={"scale": [2, 4, 16]},
                               fixed_free={"reg_weight": 0.25}, n_seeds=2)
    assert dict(report.best.config) == {"scale": 4.0}
    assert report.best.objective_mean == pytest.approx(0.17)   # (0.18+0.16)/2, red ignored
    assert report.best.n_seeds == 2
    # table is sorted best-first and every config aggregated its 2 seeds
    assert [dict(c.config)["scale"] for c in report.configs] == [4.0, 2.0, 16.0]
    assert all(c.n_seeds == 2 for c in report.configs)


def test_min_seeds_guardrail_excludes_underseeded_winner():
    # scale 2 has the highest single-seed point (+0.30) but only 1 seed; scale 4 has 3 seeds at
    # +0.16. With min_seeds=3, the noisy single-seed config must NOT win.
    cells = [
        _i2p_cell(2, 0.25, 777, 0.30),
        _i2p_cell(4, 0.25, 777, 0.17), _i2p_cell(4, 0.25, 778, 0.16), _i2p_cell(4, 0.25, 779, 0.15),
    ]
    report = search_over_cells(I2P_SPEC, cells, search_dims={"scale": [2, 4]},
                               fixed_free={"reg_weight": 0.25}, n_seeds=3, min_seeds=3)
    assert dict(report.best.config) == {"scale": 4.0}  # multi-seed config wins despite lower mean
    assert report.best_under_seeded is False
    # relaxing min_seeds to 1 lets the single-seed peak win (and flags nothing)
    relaxed = search_over_cells(I2P_SPEC, cells, search_dims={"scale": [2, 4]},
                                fixed_free={"reg_weight": 0.25}, n_seeds=3, min_seeds=1)
    assert dict(relaxed.best.config) == {"scale": 2.0}


def test_all_underseeded_falls_back_with_warning():
    cells = [_i2p_cell(2, 0.25, 777, 0.1), _i2p_cell(4, 0.25, 777, 0.2)]  # all n=1
    report = search_over_cells(I2P_SPEC, cells, search_dims={"scale": [2, 4]},
                               fixed_free={"reg_weight": 0.25}, n_seeds=3, min_seeds=3)
    assert report.best is not None and dict(report.best.config) == {"scale": 4.0}
    assert report.best_under_seeded is True  # no config met min_seeds -> flagged fallback


def test_fixed_free_filters_out_other_values():
    # reg 0.5 cells must be excluded when reg_weight is pinned to 0.25
    cells = [_i2p_cell(4, 0.25, 777, 0.18), _i2p_cell(4, 0.5, 777, 0.90)]
    report = search_over_cells(I2P_SPEC, cells, search_dims={"scale": [4]},
                               fixed_free={"reg_weight": 0.25}, n_seeds=1)
    assert report.best.objective_mean == pytest.approx(0.18)  # not the reg=0.5 cell's 0.90


def test_budget_and_partition_are_reported():
    cells = [_i2p_cell(2, 0.25, 777, 0.1), _i2p_cell(4, 0.25, 777, 0.2)]
    report = search_over_cells(I2P_SPEC, cells, search_dims={"scale": [2, 4]},
                               fixed_free={"reg_weight": 0.25}, n_seeds=3)
    d = report.to_dict()
    assert d["search_budget"] == {"n_configs": 2, "n_seeds": 3, "n_trials": 6,
                                  "search_space": {"scale": [2, 4]}}
    assert set(d["partition"]) == {FREE, SUBSTRATE, SHAPE_IDENTITY}
    assert "rank" in d["partition"][SHAPE_IDENTITY]
    assert d["objective"].startswith("ImageReward gain")


def test_default_scale_string_normalizes():
    cells = [_i2p_cell("default", 0.25, 777, 0.05)]
    cells[0]["metadata"]["lora_scale"] = "default"
    report = search_over_cells(I2P_SPEC, cells, search_dims={"scale": ["default"]},
                               fixed_free={"reg_weight": 0.25}, n_seeds=1)
    assert report.best is not None and report.best.objective_mean == pytest.approx(0.05)


# ── launch mode is restart-safe ──────────────────────────────────────────────────────────────
def test_launch_skips_existing_and_runs_missing(tmp_path):
    from adapterbench.autoresearch import _cell_dirname
    calls = []

    def fake_run(argv, out_dir):
        calls.append(dict(argv=argv, out_dir=out_dir))

    # pre-create the results.jsonl for (scale=2, seed=777) so it is skipped
    existing = tmp_path / _cell_dirname(I2P_SPEC, "lora", {"scale": 2.0, "reg_weight": 0.25}, 777)
    existing.mkdir(parents=True)
    (existing / "results.jsonl").write_text("{}\n")

    launch_missing_cells(I2P_SPEC, tmp_path, search_dims={"scale": [2.0, 4.0]},
                         fixed_free={"reg_weight": 0.25}, seeds=[777], codec="lora",
                         gpus=["0"], run=fake_run)
    # only the (scale=4) cell should have been launched; (scale=2) was skipped
    assert len(calls) == 1
    assert "scale4" in Path(calls[0]["out_dir"]).name


def test_launch_rejects_non_free_partition(tmp_path):
    with pytest.raises(ValueError, match="not FREE"):
        launch_missing_cells(I2P_SPEC, tmp_path, search_dims={"rank": [8]}, fixed_free={},
                             seeds=[777], codec="lora", gpus=["0"], run=lambda *a: None)
