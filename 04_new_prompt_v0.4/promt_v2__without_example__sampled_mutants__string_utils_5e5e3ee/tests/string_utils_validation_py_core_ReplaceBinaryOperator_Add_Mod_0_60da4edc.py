from string_utils.validation import is_isbn_10

def test__is_isbn_10_mutant_killing_invalid():
    """
    Test that the function correctly identifies an invalid ISBN 10 number.
    The input '1234567891' should return false on the baseline,
    but true on the mutant due to the faulty logic.
    """
    output = is_isbn_10('1234567891')
    assert output == False  # This should pass for baseline but fail for mutant