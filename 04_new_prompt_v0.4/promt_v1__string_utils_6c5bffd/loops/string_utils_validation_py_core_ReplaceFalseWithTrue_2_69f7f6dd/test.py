from string_utils.validation import is_url

def test__is_url_mutant_killing():
    """
    Test the is_url function with an empty string input.
    The baseline should return False for an empty string, 
    while the mutant should return True, thereby killing the mutant.
    """
    # Test case expecting Baseline to return False and Mutant to return True
    result = is_url('')
    assert result is False, f"Expected False, but got {result}."