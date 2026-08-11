# Agent orientation

This is **AdapterBench**: it tests whether the *shape* of a hypernetwork-generated PEFT
adapter matters under live end-to-end language-model SFT. It has exactly two active
settings: task-description conditioning (T2A, gemma-2-2b) and document conditioning
(D2A, Qwen3-0.6B). LoRA is the baseline codec; new representations are added and
evaluated one at a time across both settings (registered so far: lora, ia3, lokr,
fourierft, steering).

The core benchmark implementation and the LoRA/(IA)³/LoKr/FourierFT reference results
are complete. The current project phase is **codec exploration**: evaluate each newly
registered codec in both settings under `AUTORESEARCH.md`'s protocol, keeping every
displayed result derived from canonical artifacts (leaderboards, paper, website, and
commands must agree — `adapterbench results check`). New *settings* are out of scope.

**Start here: [`PROJECT_PLAN.md`](PROJECT_PLAN.md)** — current status, architecture, how
to run the benchmark, results, and hard-won gotchas.

For day-to-day commands (environment setup, running the benchmark), see
[`SETUP.md`](SETUP.md). For landing a new codec via pull request, see
[`CONTRIBUTING.md`](CONTRIBUTING.md). For the plugin contract every backend/evaluator implements, see
[`BENCHMARK_CONTRACT.md`](BENCHMARK_CONTRACT.md).
