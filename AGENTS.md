# Agent orientation

This is **AdapterBench**: it tests whether the *shape* of a hypernetwork-generated PEFT
adapter matters (LoRA, FreezeALoRA, LoKr, FourierFT, IA3, activation steering), across two
settings: (1) disk-artifact reproduction of released Text-to-LoRA checkpoints (Gemma,
Mistral, Llama) scored via vLLM, and (2) live end-to-end SFT with a hypernetwork trained
entirely from scratch against a real Qwen3-0.6B interpreter, comparing all six
representations head to head. Doc-to-LoRA integration is planned but not yet started.

**Start here: [`PROJECT_PLAN.md`](PROJECT_PLAN.md)** — current status, architecture, how
to run each setting, results, and hard-won gotchas.

For day-to-day commands (environment setup, running the benchmark, the cross-venv
generation bridge), see [`SETUP.md`](SETUP.md). For the plugin contract every backend/
evaluator implements, see [`BENCHMARK_CONTRACT.md`](BENCHMARK_CONTRACT.md).
