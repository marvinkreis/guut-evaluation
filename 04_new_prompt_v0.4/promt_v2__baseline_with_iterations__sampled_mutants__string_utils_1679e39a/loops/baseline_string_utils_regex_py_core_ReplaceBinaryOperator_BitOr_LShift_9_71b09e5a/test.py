from string_utils._regex import PRETTIFY_RE

def test__UPPERCASE_AFTER_SIGN():
    """
    Test whether the regex correctly identifies when a character that must be followed by an uppercase letter is present.
    The input here is a sentence that follows the correct format of having a period followed by a space and an uppercase letter,
    as expected by the regex. The mutant changes the regex by incorrectly using a bitwise shift operation instead of the logical
    OR, which will alter the behavior of the regex.
    """
    test_string = "This is a test. A new beginning."
    output = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].findall(test_string)
    assert output == [('. A')]