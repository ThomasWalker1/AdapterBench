# Agent orientation

This is **AdapterBench**: it tests whether the *shape* of a hypernetwork-generated PEFT
adapter matters under live end-to-end language-model SFT. It has exactly two active
settings: task-description conditioning (T2L, gemma-2-2b) and document conditioning
(D2L, Qwen3-0.6B). LoRA is the current baseline codec; new representations are added and
evaluated one at a time across both settings.

**Start here: [`PROJECT_PLAN.md`](PROJECT_PLAN.md)** — current status, architecture, how
to run the benchmark, results, and hard-won gotchas.

For day-to-day commands (environment setup, running the benchmark), see
[`SETUP.md`](SETUP.md). For the plugin contract every backend/evaluator implements, see
[`BENCHMARK_CONTRACT.md`](BENCHMARK_CONTRACT.md).
