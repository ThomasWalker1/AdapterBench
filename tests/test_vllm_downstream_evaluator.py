from pathlib import Path

from adapterbench.contracts import AdapterArtifact, TaskExample
from adapterbench.vllm_downstream_evaluator import VLLMDownstreamEvaluator


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return f"<chat>{messages[0]['content']}</chat>"


def _make_evaluator(use_icl=False):
    # Bypass __init__ (which loads a real tokenizer and checks for the upstream venv) -
    # these tests exercise the pure grouping/prompting/scoring logic only.
    evaluator = object.__new__(VLLMDownstreamEvaluator)
    evaluator.model_id = "fake/model"
    evaluator.trial_id = "trial-1"
    evaluator.device = "cuda:0"
    evaluator.max_new_tokens = 512
    evaluator.gpu_memory_utilization = 0.85
    evaluator.tokenizer = FakeTokenizer()
    from adapterbench.hf_downstream_evaluator import build_prefill_by_family

    evaluator._prefill_by_family = build_prefill_by_family(use_icl)
    return evaluator


def _example(family, task_id, target_text):
    return TaskExample(
        task_id=task_id,
        condition="do the task",
        input_text=f"input for {task_id}",
        target_text=target_text,
        family=family,
        metadata={"condition_variant": 0},
    )


def test_prompts_for_renders_chat_template_and_family_prefill():
    evaluator = _make_evaluator(use_icl=False)
    examples = [_example("gsm8k", "gsm8k::0", "42")]
    prompts = evaluator._prompts_for("gsm8k", examples)
    assert prompts == ["<chat>input for gsm8k::0</chat>Let's think step by step."]


def test_prompts_for_has_no_prefill_for_choice_families_without_icl():
    evaluator = _make_evaluator(use_icl=False)
    examples = [_example("arc_easy", "arc_easy::0", "A")]
    prompts = evaluator._prompts_for("arc_easy", examples)
    assert prompts == ["<chat>input for arc_easy::0</chat>"]


def test_score_dispatches_by_family():
    assert VLLMDownstreamEvaluator._score("gsm8k", "the answer is 42", "42") is True
    assert VLLMDownstreamEvaluator._score("boolq", "yes", "true") is True
    assert VLLMDownstreamEvaluator._score("arc_easy", "B: because", "B") is True


def test_iter_evaluate_builds_one_result_per_family_with_adapter_metadata(monkeypatch):
    evaluator = _make_evaluator(use_icl=False)
    examples = [_example("boolq", "boolq::0", "true"), _example("boolq", "boolq::1", "false")]
    artifact = AdapterArtifact(
        task_id="boolq",
        adapter="lora",
        path=Path("/fake/adapters/boolq"),
        format="peft",
        generated_parameter_count=123,
        generation_seconds=0.5,
        metadata={"condition": "do the task"},
    )

    def fake_stream_groups(self, groups):
        assert len(groups) == 1
        assert groups[0]["adapter_dir"] == str(artifact.path)
        yield "boolq", ["yes", "no"], 1.23

    monkeypatch.setattr(VLLMDownstreamEvaluator, "_stream_groups", fake_stream_groups)

    results = evaluator.evaluate({"boolq": artifact}, examples, split="test")
    assert len(results) == 1
    result = results[0]
    assert result.adapter == "lora"
    assert result.task_id == "boolq"
    assert result.metrics == {"accuracy": 1.0, "n_examples": 2.0}
    assert result.generated_parameter_count == 123
    assert result.inference_seconds == 1.23
    assert result.metadata["backend"] == "vllm"
    assert result.metadata["artifact_metadata"] == {"condition": "do the task"}


def test_iter_evaluate_frozen_passes_no_adapter_dir(monkeypatch):
    evaluator = _make_evaluator(use_icl=False)
    examples = [_example("gsm8k", "gsm8k::0", "10")]

    def fake_stream_groups(self, groups):
        assert groups[0]["adapter_dir"] is None
        yield "gsm8k", ["the answer is 10"], 0.1

    monkeypatch.setattr(VLLMDownstreamEvaluator, "_stream_groups", fake_stream_groups)

    results = evaluator.evaluate_frozen(examples, split="test")
    assert len(results) == 1
    assert results[0].adapter == "frozen_interpreter"
    assert results[0].metrics == {"exact_match": 1.0, "n_examples": 1.0}
    assert results[0].generated_parameter_count == 0
