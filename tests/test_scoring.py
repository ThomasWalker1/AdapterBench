from adapterbench.scoring import (
    get_binary_accuracy,
    get_bool_value,
    get_choice,
    get_choice_accuracy,
    get_gsm8k_accuracy,
)


def test_get_choice_extracts_leading_letter_ignoring_case_and_punctuation():
    assert get_choice("B") == "b"
    assert get_choice(" b: because reasons") == "b"
    assert get_choice("`C`") == "c"
    assert get_choice("(D) obviously") == "d"


def test_get_choice_extracts_leading_digit_for_number_labeled_choices():
    assert get_choice("2") == "2"
    assert get_choice("  3: some ending text") == "3"


def test_get_choice_returns_none_when_nothing_matches():
    assert get_choice("   ") is None
    assert get_choice("...") is None


def test_get_choice_accuracy_compares_extracted_choices_not_raw_text():
    assert get_choice_accuracy("B: because bacteria", "B") is True
    # "The answer is B" starts with the letter "t" ("The"), which get_choice mistakes
    # for choice "t" - a real limitation this test locks in since it's inherited
    # verbatim from upstream, not something we should silently "fix" and diverge on.
    assert get_choice_accuracy("The answer is B", "B") is False
    assert get_choice_accuracy("A", "B") is False


def test_get_bool_value_prefers_digits_over_words():
    assert get_bool_value("1") is True
    assert get_bool_value("0") is False
    assert get_bool_value("true") is True
    assert get_bool_value("False") is False
    assert get_bool_value("yes") is True
    assert get_bool_value("no, that's wrong") is False
    assert get_bool_value("unrelated text") is None


def test_get_binary_accuracy_matches_loose_keyword_across_representations():
    assert get_binary_accuracy("True, because...", "true") is True
    assert get_binary_accuracy("no", "false") is True
    assert get_binary_accuracy("yes definitely", "false") is False


def test_get_binary_accuracy_is_false_when_either_side_is_unparseable():
    assert get_binary_accuracy("I'm not sure", "true") is False


def test_get_gsm8k_accuracy_uses_last_number_in_generated_text():
    assert get_gsm8k_accuracy("blah 12 more blah 42", "42") is True
    assert get_gsm8k_accuracy("The answer is 1,234.", "1234") is True
    assert get_gsm8k_accuracy("no numbers here", "42") is False
