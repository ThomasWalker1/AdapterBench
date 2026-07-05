# Archive: retired reconstruction-matching code

Superseded when the benchmark was restructured to train solely via live end-to-end SFT
(hypernetwork output hooked into a real frozen-interpreter forward pass, trained on real
next-token loss) rather than parameter-space reconstruction matching. See
`PROJECT_PLAN.md`'s Phase 3 (reconstruction pilot, real results recorded there) and Phase
4 (the SFT restructuring that replaced it) for the full history — this code isn't deleted
because this project has no git history of its own to recover it from otherwise.

Not wired into the installed `peft_hnet` package or the test suite (`pyproject.toml`'s
`testpaths = ["tests"]` deliberately excludes this directory) — the modules here import
each other by their old relative paths and won't run without being copied back into
`src/peft_hnet/`.

- `src/peft_hnet/t2p/pilot.py`, `t2p/oracle_targets.py` — the dense-ΔW leave-one-task-out
  reconstruction pilot (Phase 3), validated against real Mistral-7B oracle LoRAs.
- `src/peft_hnet/generators.py`, `reconstruction.py` — an earlier, simpler flat-vector
  reconstruction training loop (`BasisStateGenerator`), never connected to `t2p/`.
- `tests/test_pilot.py`, `test_oracle_targets.py`, `test_generators.py` — their
  corresponding regression tests (the `AdapterStateLayout` test that used to share a file
  with `test_generators.py` was split out to `tests/test_state.py`, which stays active —
  `state.py` is unrelated to reconstruction and is still used by `hf_smoke.py`).
