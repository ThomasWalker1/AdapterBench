from peft_hnet.task_examples import (
    _arc_examples,
    _boolq_examples,
    _gsm8k_examples,
    _hellaswag_examples,
    build_conditions,
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
    assert example.target_text == "Blue"
    assert example.metadata["answer_index"] == 0
    assert example.metadata["choices"] == ["Blue", "Green"]


def test_arc_examples_respects_limit():
    examples = _arc_examples("arc_easy", "c", ARC_ROWS * 5, limit=2, variant=0)
    assert len(examples) == 2


def test_boolq_examples_maps_bool_answer_to_yes_no_choice():
    examples = _boolq_examples("boolq", "c", BOOLQ_ROWS, limit=10, variant=0)
    assert len(examples) == 1
    assert examples[0].target_text == "yes"
    assert examples[0].metadata["answer_index"] == 1
    assert "Question:" in examples[0].input_text


def test_hellaswag_examples_skip_rows_with_unparseable_label():
    examples = _hellaswag_examples("hellaswag", "c", HELLASWAG_ROWS, limit=10, variant=0)
    assert len(examples) == 1
    assert examples[0].target_text == "walked in."


def test_gsm8k_examples_extract_final_numeric_answer():
    examples = _gsm8k_examples("gsm8k", "c", GSM8K_ROWS, limit=10, variant=0)
    assert len(examples) == 1
    assert examples[0].target_text == "4"


def test_build_conditions_selects_variant_by_index():
    descriptions = {"arc_easy": ["first", "second"], "boolq": ["alpha", "beta"]}
    conditions = build_conditions(descriptions, ["arc_easy", "boolq"], variant=1)
    assert conditions == {"arc_easy": "second", "boolq": "beta"}


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
