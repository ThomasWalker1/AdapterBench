# Agent orientation

This is **AdapterBench**: it benchmarks PEFT representations (LoRA, FourierFT, KronA,
IA3, activation steering, prefix-tuning) as **hypernetwork output targets**, generated
live and hooked into a frozen interpreter's forward pass for end-to-end SFT training —
generalizing the mechanism behind Sakana AI's Text-to-LoRA beyond LoRA itself. Also hosts
earlier disk-artifact-based settings (a released Text-to-LoRA checkpoint, a self-trained
reconstruction hypernetwork, PAW/FuzzyBench metadata; Doc-to-LoRA integration is planned
but not yet started).

**Start here: [`PROJECT_PLAN.md`](PROJECT_PLAN.md)** — current status, what's actually
implemented vs. still a gap, how to check/resume the latest run, hard-won gotchas, and
the prioritized roadmap (Phase 2/3/4).

For day-to-day commands (environment setup, running the benchmark, the cross-venv
generation bridge), see [`SETUP.md`](SETUP.md). For the plugin contract every backend/
evaluator implements, see [`BENCHMARK_CONTRACT.md`](BENCHMARK_CONTRACT.md).
