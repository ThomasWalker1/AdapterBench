import pytest
import torch
import yaml
from safetensors.torch import save_file

from peft_hnet.t2p.oracle_targets import (
    build_task_oracle_targets,
    find_oracle_adapter_paths,
    load_lora_delta_targets,
)


def _write_fake_lora(path, num_layers, rank, in_features, out_features):
    a_list, b_list, state_dict = [], [], {}
    for layer in range(num_layers):
        a = torch.randn(rank, in_features)
        b = torch.randn(out_features, rank)
        a_list.append(a)
        b_list.append(b)
        state_dict[f"base_model.model.model.layers.{layer}.self_attn.q_proj.lora_A.weight"] = a
        state_dict[f"base_model.model.model.layers.{layer}.self_attn.q_proj.lora_B.weight"] = b
    save_file(state_dict, str(path))
    return a_list, b_list


def test_load_lora_delta_targets_matches_manual_bmm(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "adapter_model.safetensors"
    a_list, b_list = _write_fake_lora(path, num_layers=2, rank=4, in_features=6, out_features=5)

    targets = load_lora_delta_targets(path, ["q_proj"], num_layers=2)

    expected = torch.stack([b @ a for a, b in zip(a_list, b_list)], dim=0)
    assert targets["q_proj"].shape == (2, 5, 6)
    torch.testing.assert_close(targets["q_proj"], expected)


def test_load_lora_delta_targets_raises_on_missing_layer(tmp_path):
    path = tmp_path / "adapter_model.safetensors"
    _write_fake_lora(path, num_layers=2, rank=4, in_features=6, out_features=5)

    with pytest.raises(ValueError, match="missing lora_A"):
        load_lora_delta_targets(path, ["q_proj"], num_layers=3)


def test_build_task_oracle_targets_maps_each_task(tmp_path):
    torch.manual_seed(0)
    path_a = tmp_path / "task_a.safetensors"
    path_b = tmp_path / "task_b.safetensors"
    _write_fake_lora(path_a, num_layers=1, rank=2, in_features=4, out_features=4)
    _write_fake_lora(path_b, num_layers=1, rank=2, in_features=4, out_features=4)

    targets = build_task_oracle_targets({"task_a": path_a, "task_b": path_b}, ["q_proj"], num_layers=1)

    assert set(targets) == {"task_a", "task_b"}
    assert targets["task_a"]["q_proj"].shape == (1, 4, 4)


def _write_oracle_run(root, run_name, train_ds_names):
    run_dir = root / run_name
    run_dir.mkdir()
    (run_dir / "args.yaml").write_text(yaml.dump({"train_ds_names": train_ds_names}))
    (run_dir / "adapter_model.safetensors").write_bytes(b"")
    return run_dir


def test_find_oracle_adapter_paths_maps_task_ids_to_run_dirs(tmp_path):
    _write_oracle_run(tmp_path, "20260101-000000_aaa", ["lol_001"])
    _write_oracle_run(tmp_path, "20260101-000001_bbb", ["lol_002"])
    _write_oracle_run(tmp_path, "20260101-000002_ccc", ["lol_001", "lol_002"])  # multi-task, should be skipped

    found = find_oracle_adapter_paths(tmp_path, ["lol_001", "lol_002"])

    assert found["lol_001"] == tmp_path / "20260101-000000_aaa" / "adapter_model.safetensors"
    assert found["lol_002"] == tmp_path / "20260101-000001_bbb" / "adapter_model.safetensors"


def test_find_oracle_adapter_paths_raises_when_task_missing(tmp_path):
    _write_oracle_run(tmp_path, "20260101-000000_aaa", ["lol_001"])

    with pytest.raises(FileNotFoundError, match="lol_002"):
        find_oracle_adapter_paths(tmp_path, ["lol_001", "lol_002"])
