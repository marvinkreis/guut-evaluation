from string_utils._regex import HTML_RE

def test__html_re_regex():
    """
    Test to check whether HTML_RE regex compiles without errors.
    The baseline code compiles the regex correctly, while the mutant will fail due to a TypeError caused by an incorrect operator.
    """
    try:
        compiled_regex = HTML_RE.pattern
        assert compiled_regex is not None
    except Exception as e:
        assert False, f"An error occurred: {e}"

test__html_re_regex()