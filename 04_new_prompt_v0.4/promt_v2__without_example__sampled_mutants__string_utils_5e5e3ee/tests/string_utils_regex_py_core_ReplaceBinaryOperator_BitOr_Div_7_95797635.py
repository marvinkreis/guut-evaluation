from string_utils._regex import PRETTIFY_RE

def test__prettify_duplicates():
    """
    Test whether the original 'DUPLICATES' regex can correctly handle a test string containing repeated whitespace.
    The input contains repeated characters (punctuation or whitespace) where the baseline should execute successfully.
    The mutant code should fail to run due to a syntax error in the regex compilation.
    """
    test_string = "!!! This is a test...??  "

    baseline_matches = PRETTIFY_RE['DUPLICATES'].findall(test_string)
    print(f"Baseline matches: {baseline_matches}")

    # Asserting that some matches are found in the baseline
    assert len(baseline_matches) > 0, "The baseline should find matches."

    # The mutant should fail to reach this point due to a compilation error