# Agent orientation

This is **AdapterBench**: it tests whether the *shape* of a hypernetwork-generated PEFT
adapter matters (LoRA, FreezeALoRA, LoKr, FourierFT, IA3, activation steering) under live
end-to-end SFT — a hypernetwork trained entirely from scratch against a real Qwen3-0.6B
interpreter, comparing all six representations head to head.

**Start here: [`PROJECT_PLAN.md`](PROJECT_PLAN.md)** — current status, architecture, how
to run the benchmark, results, and hard-won gotchas.

For day-to-day commands (environment setup, running the benchmark), see
[`SETUP.md`](SETUP.md). For the plugin contract every backend/evaluator implements, see
[`BENCHMARK_CONTRACT.md`](BENCHMARK_CONTRACT.md).
