from string_utils.validation import is_isogram

def test__is_isogram_mutant_killer():
    """
    Test to check that the function returns false for non-isogram strings.
    The input 'hello' has repeated characters, so it should not be classified as an isogram.
    The test will pass for the baseline and fail for the mutant.
    """
    output = is_isogram('hello')
    assert output is False, f"Expected False, got {output}"