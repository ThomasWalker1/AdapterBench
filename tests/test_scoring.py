import math

from adapterbench.scoring import (
    get_binary_accuracy,
    get_bool_value,
    get_choice,
    get_choice_accuracy,
    get_gsm8k_accuracy,
    lcs_length,
    rouge_l,
    rouge_l_max,
    rouge_tokenize,
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


def test_rouge_tokenize_lowercases_and_splits_on_every_non_alphanumeric_run():
    assert rouge_tokenize("Hello, World!  It's 42.") == ["hello", "world", "it", "s", "42"]
    assert rouge_tokenize("") == []
    assert rouge_tokenize("---   ...") == []


def test_rouge_tokenize_does_not_stem_because_sni_sets_use_stemmer_false():
    assert rouge_tokenize("running runs") == ["running", "runs"]
    # Consequence of no stemming: an inflected match scores zero, not partial credit.
    assert rouge_l("running", "run") == 0.0


def test_lcs_length_is_subsequence_not_substring_and_respects_order():
    assert lcs_length(["a", "b", "c"], ["a", "x", "b", "y", "c"]) == 3
    assert lcs_length(["a", "b"], ["b", "a"]) == 1
    assert lcs_length([], ["a"]) == 0
    assert lcs_length(["a"], []) == 0


def test_rouge_l_is_one_for_an_exact_match_up_to_case_and_punctuation():
    assert rouge_l("the cat sat", "the cat sat") == 1.0
    assert rouge_l("The cat, sat.", "the cat sat") == 1.0


def test_rouge_l_is_zero_when_no_token_is_shared_or_either_side_is_empty():
    assert rouge_l("alpha beta", "gamma delta") == 0.0
    assert rouge_l("", "gamma") == 0.0
    assert rouge_l("gamma", "") == 0.0
    assert rouge_l("!!!", "gamma") == 0.0


def test_rouge_l_f_measure_penalizes_a_generation_that_rambles_past_the_answer():
    # LCS = 3 ("the cat sat"), precision 3/6, recall 3/3 -> F = 2/3. This precision
    # penalty is why the 32-new-token cap matters: an over-long decode loses ROUGE-L
    # even when it contains the gold answer, unlike this repo's gold-in-generation EM.
    assert math.isclose(rouge_l("the cat sat on the mat", "the cat sat"), 2.0 / 3.0)


def test_rouge_l_gives_partial_credit_where_exact_match_gives_none():
    # gold ["positive","sentiment"] vs gen ["sentiment","is","positive"]: order-respecting
    # LCS is 1, precision 1/3, recall 1/2 -> F = 0.4.
    assert math.isclose(rouge_l("sentiment is positive", "positive sentiment"), 0.4)
    assert rouge_l("cat the", "the cat") == 0.5


def test_rouge_l_max_keeps_the_best_reference_as_sni_does():
    assert rouge_l_max("the cat sat", ["a dog barked", "the cat sat"]) == 1.0
    assert rouge_l_max("the cat sat", []) == 0.0
