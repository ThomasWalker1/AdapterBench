"""Unit tests for the derived, provenance-stamped leaderboard (CPU, no GPU)."""

from __future__ import annotations

from adapterbench.leaderboard import (
    LeaderboardRecord, append_record, load_records, record_from_autoresearch, render_leaderboard,
)


def _rec(codec="lora", mean=0.16, sha="abc123def456", gen=32768, split="eval", budget=None):
    return LeaderboardRecord(
        setting="i2p_hypernoise", codec=codec, main_git_sha=sha, trial_id=f"t2p--{codec}--xyz",
        objective_name="ImageReward gain", objective_mean=mean, objective_std=0.015,
        best_config={"scale": 4.0}, seeds=[777, 778, 779], data_split=split,
        search_budget=budget or {"n_configs": 7, "n_seeds": 3}, generated_output_size=gen,
    )


def test_append_is_idempotent_on_provenance_key(tmp_path):
    append_record(_rec(), tmp_path)
    append_record(_rec(), tmp_path)  # identical provenance -> no duplicate row
    assert len(load_records(tmp_path)) == 1


def test_new_sha_appends_a_new_row(tmp_path):
    append_record(_rec(sha="sha_one______"), tmp_path)
    append_record(_rec(sha="sha_two______"), tmp_path)  # main moved -> new row, old kept
    assert len(load_records(tmp_path)) == 2


def test_render_groups_and_keeps_losers(tmp_path):
    append_record(_rec(codec="lora", mean=0.16, gen=32768), tmp_path)
    append_record(_rec(codec="loser_shape", mean=-0.05, gen=65536), tmp_path)  # a loser stays
    board = render_leaderboard(tmp_path)
    assert "i2p_hypernoise" in board
    assert "lora" in board and "loser_shape" in board  # losers are NOT curated away
    # winner (higher objective) is ranked above the loser
    assert board.index("lora ") < board.index("loser_shape")


def test_render_at_sha_snapshot(tmp_path):
    append_record(_rec(sha="sha_old______", mean=0.10), tmp_path)
    append_record(_rec(sha="sha_new______", mean=0.16), tmp_path)
    snap = render_leaderboard(tmp_path, at_sha="sha_old______")
    assert "sha_old" in snap and "sha_new" not in snap


def test_record_from_autoresearch_report():
    report = {
        "setting": "i2p_hypernoise", "codec": "lora", "objective": "ImageReward gain",
        "best_config": {"scale": 4.0}, "best_objective_mean": 0.16, "best_objective_std": 0.015,
        "search_budget": {"n_configs": 7, "n_seeds": 3}, "best_under_seeded": False,
    }
    rec = record_from_autoresearch(report, main_git_sha="deadbeef1234", trial_id="t--lora--x",
                                   data_split="eval_prompts", generated_output_size=32768)
    assert rec.main_git_sha == "deadbeef1234" and rec.objective_mean == 0.16
    assert rec.generated_output_size == 32768


def test_record_from_autoresearch_requires_a_best():
    import pytest
    with pytest.raises(ValueError, match="no best config"):
        record_from_autoresearch({"best_config": None}, main_git_sha="x", trial_id="t",
                                 data_split="s")
