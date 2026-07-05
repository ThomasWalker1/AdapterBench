import torch

from peft_hnet.t2p.hypernetwork import TextToPeftHypernetwork
from peft_hnet.t2p.pilot import run_leave_one_out_pilot


def _synthetic_problem(seed=0):
    torch.manual_seed(seed)
    module_shapes = {"q_proj": (8, 8), "v_proj": (8, 4)}
    num_layers = 2
    task_ids = ["t0", "t1", "t2"]
    condition_embeddings = {task_id: torch.randn(6) for task_id in task_ids}
    oracle_targets = {
        task_id: {
            name: torch.randn(num_layers, out_features, in_features) * 0.01
            for name, (in_features, out_features) in module_shapes.items()
        }
        for task_id in task_ids
    }
    return module_shapes, num_layers, task_ids, condition_embeddings, oracle_targets


def test_leave_one_out_pilot_runs_one_fold_per_task_with_finite_losses():
    module_shapes, num_layers, task_ids, condition_embeddings, oracle_targets = _synthetic_problem()

    def factory():
        return TextToPeftHypernetwork(
            condition_dim=6,
            module_shapes=module_shapes,
            num_layers=num_layers,
            representation="fourierft",
            latent_dim=16,
            head_dim=16,
            n_frequency=8,
            seed=1,
        )

    stats = run_leave_one_out_pilot(
        factory, task_ids, condition_embeddings, oracle_targets, steps=5, learning_rate=1e-2, delta_w_scaling=100.0
    )

    assert [fold.held_out_task_id for fold in stats] == task_ids
    for fold in stats:
        assert fold.held_out_task_id not in fold.train_task_ids
        assert set(fold.train_task_ids) == set(task_ids) - {fold.held_out_task_id}
        assert fold.steps == 5
        for value in (fold.initial_train_loss, fold.final_train_loss, fold.held_out_loss, fold.zero_baseline_loss):
            assert torch.isfinite(torch.tensor(value))


def test_leave_one_out_pilot_can_compute_a_subset_of_folds():
    """Supports splitting independent folds across parallel processes/GPUs: a caller
    should be able to request just one held-out task's fold, still trained against the
    full task universe's other members, and get exactly that one fold back."""
    module_shapes, num_layers, task_ids, condition_embeddings, oracle_targets = _synthetic_problem()

    def factory():
        return TextToPeftHypernetwork(
            condition_dim=6,
            module_shapes=module_shapes,
            num_layers=num_layers,
            representation="fourierft",
            latent_dim=16,
            head_dim=16,
            n_frequency=8,
            seed=1,
        )

    stats = run_leave_one_out_pilot(
        factory,
        task_ids,
        condition_embeddings,
        oracle_targets,
        steps=5,
        learning_rate=1e-2,
        delta_w_scaling=100.0,
        held_out_task_ids=["t1"],
    )

    assert [fold.held_out_task_id for fold in stats] == ["t1"]
    assert set(stats[0].train_task_ids) == {"t0", "t2"}


def test_leave_one_out_pilot_beats_zero_baseline_when_tasks_share_signal():
    """Sanity check: if every task's oracle target is (near-)identical, the hypernetwork
    should learn to reproduce it and clearly beat predicting zero."""
    torch.manual_seed(0)
    module_shapes = {"q_proj": (8, 8)}
    num_layers = 1
    task_ids = ["t0", "t1", "t2"]
    shared_target = torch.randn(num_layers, 8, 8) * 0.1
    condition_embeddings = {task_id: torch.randn(6) for task_id in task_ids}
    oracle_targets = {task_id: {"q_proj": shared_target.clone()} for task_id in task_ids}

    def factory():
        return TextToPeftHypernetwork(
            condition_dim=6,
            module_shapes=module_shapes,
            num_layers=num_layers,
            representation="fourierft",
            latent_dim=16,
            head_dim=16,
            n_frequency=16,
            seed=1,
        )

    stats = run_leave_one_out_pilot(
        factory, task_ids, condition_embeddings, oracle_targets, steps=300, learning_rate=1e-2, delta_w_scaling=100.0
    )

    for fold in stats:
        assert fold.held_out_loss < fold.zero_baseline_loss
