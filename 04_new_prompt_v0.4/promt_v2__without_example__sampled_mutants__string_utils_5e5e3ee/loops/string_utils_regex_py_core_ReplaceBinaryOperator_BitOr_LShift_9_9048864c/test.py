from string_utils._regex import PRETTIFY_RE

def test__uppercase_after_sign():
    """
    Test the 'UPPERCASE_AFTER_SIGN' regex against strings with punctuation followed by uppercase letters.
    The baseline should compile and provide expected results, while the mutant will raise an OverflowError.
    """
    test_strings = ["Hello.", "See?"]
    results = {s: bool(PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(s)) for s in test_strings}
    assert results['Hello.'] == False
    assert results['See?'] == False