from string_utils._regex import PRETTIFY_RE

def test__prettify_uppercase_after_sign():
    """
    Test that the regex for matching characters that must be followed by uppercase letters
    correctly identifies instances where characters (like '.', '?', etc.) precede uppercase letters.
    The mutant modifies the regex by replacing the OR operator with a bitwise operator,
    which causes incorrect matching behavior. For example, 'hello. World' should match,
    but the mutant would fail to recognize the uppercase 'W' following '.'.
    """
    test_string = "hello. World"
    output = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].search(test_string)
    assert output is not None