from string_utils._regex import PRETTIFY_RE

def test__upper_case_after_sign():
    """
    Test the behavior of the regex for matching uppercase letters after punctuation marks.
    The baseline should return matches for uppercase letters after punctuation,
    while the mutant should raise an OverflowError due to incorrect regex compilation.
    """
    input_string = ".\nUppercase"
    output = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].findall(input_string)
    assert output == ['.\nU']