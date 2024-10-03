from string_utils.validation import is_isbn

def test__is_isbn():
    """
    Test the is_isbn function with input that includes hyphens.
    The baseline should return True for '978-3-16-148410-0' due to normalization being True by default,
    while the mutant should return False as it doesn't normalize hyphens.
    This test effectively distinguishes between the baseline and mutant implementations.
    """
    result = is_isbn('978-3-16-148410-0')
    assert result is True, f"Expected True for the baseline, but got {result}"