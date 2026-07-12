from adapterbench.task_examples import (
    _arc_examples,
    _boolq_examples,
    _gsm8k_examples,
    _hellaswag_examples,
    load_task_descriptions,
)


ARC_ROWS = [
    {
        "question": "What color is the sky?",
        "choices": {"text": ["Blue", "Green"], "label": ["A", "B"]},
        "answerKey": "A",
    },
    {
        "question": "Bad row with unknown answer key",
        "choices": {"text": ["X", "Y"], "label": ["A", "B"]},
        "answerKey": "C",
    },
]

BOOLQ_ROWS = [
    {"passage": "Water boils at 100C.", "question": "Does water boil at 100C?", "answer": True},
]

HELLASWAG_ROWS = [
    {"ctx": "She opened the door and", "endings": ["walked in.", "flew away.", "sat down.", "sang."], "label": "0"},
    {"ctx": "Bad row", "endings": ["a", "b"], "label": ""},
]

GSM8K_ROWS = [
    {"question": "What is 2+2?", "answer": "Reasoning...\n#### 4"},
    {"question": "No answer marker", "answer": "unparseable"},
]


def test_arc_examples_skip_unmatched_answer_key():
    examples = _arc_examples("arc_easy", "condition text", ARC_ROWS, limit=10, variant=0)
    assert len(examples) == 1
    example = examples[0]
    assert example.task_id == "arc_easy::0"
    assert example.condition == "condition text"
    # target_text is the raw answer key (matching upstream's response_field="answerKey"),
    # not the choice text: scoring extracts a leading letter from the model's own
    # generated answer, it doesn't rank choice log-likelihoods.
    assert example.target_text == "A"
    assert example.metadata["answer_index"] == 0
    # Padded to 4 choices with "N/A" filler, matching upstream's ARC preprocessing
    # (the template hardcodes 4 slots; this row only has 2 real choices).
    assert example.metadata["choices"] == ["Blue", "Green", "N/A", "N/A"]
    assert "A: Blue" in example.input_text and "B: Green" in example.input_text


def test_arc_examples_respects_limit():
    examples = _arc_examples("arc_easy", "c", ARC_ROWS * 5, limit=2, variant=0)
    assert len(examples) == 2


def test_boolq_examples_maps_bool_answer_to_true_false_target():
    examples = _boolq_examples("boolq", "c", BOOLQ_ROWS, limit=10, variant=0)
    assert len(examples) == 1
    assert examples[0].target_text == "true"
    assert examples[0].metadata["answer_index"] == 1
    assert "Question:" in examples[0].input_text


def test_hellaswag_examples_skip_rows_with_unparseable_label():
    examples = _hellaswag_examples("hellaswag", "c", HELLASWAG_ROWS, limit=10, variant=0)
    assert len(examples) == 1
    # target_text is the ending's index as a string (matching the template's 0/1/2/3
    # labeling), not the ending text itself.
    assert examples[0].target_text == "0"
    assert "walked in." in examples[0].input_text


def test_gsm8k_examples_extract_final_numeric_answer():
    examples = _gsm8k_examples("gsm8k", "c", GSM8K_ROWS, limit=10, variant=0)
    assert len(examples) == 1
    assert examples[0].target_text == "4"


def test_load_task_descriptions_reads_eval_ds_info(tmp_path):
    args_yaml = tmp_path / "args.yaml"
    args_yaml.write_text(
        "eval_ds_info:\n"
        "  arc_easy:\n"
        "    descriptions:\n"
        "      - first description\n"
        "      - second description\n"
    )
    descriptions = load_task_descriptions(args_yaml)
    assert descriptions == {"arc_easy": ["first description", "second description"]}
