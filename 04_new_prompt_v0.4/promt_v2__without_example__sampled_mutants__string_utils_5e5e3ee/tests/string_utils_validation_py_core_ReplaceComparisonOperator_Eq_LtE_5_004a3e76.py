from string_utils.validation import is_isogram

def test__is_isogram_mutant_killer():
    """
    This test checks the `is_isogram` function with a specific non-isogram input 'hello'.
    The baseline should return False, while the mutant should incorrectly return True.
    This difference will kill the mutant.
    """
    output = is_isogram('hello')
    assert output == False  # Expecting False for the baseline