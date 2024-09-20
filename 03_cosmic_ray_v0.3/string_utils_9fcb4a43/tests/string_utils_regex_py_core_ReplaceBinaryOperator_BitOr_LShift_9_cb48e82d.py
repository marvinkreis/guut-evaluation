from string_utils._regex import PRETTIFY_RE

def test__uppercase_after_sign():
    """The mutant code causes an OverflowError due to an invalid regex definition."""
    test_string = "Hello! How are you? I hope you're well."
    correct_result = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].findall(test_string)
    assert correct_result is not None, "The correct regex must return a result."