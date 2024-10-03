from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_re():
    """
    Test that the HTML_TAG_ONLY_RE regex compiles successfully.
    The baseline should compile this regex without raising a SyntaxError,
    whereas the mutant should fail with a TypeError due to invalid operator usage.
    """
    try:
        output = HTML_TAG_ONLY_RE.pattern
        assert True  # Expecting successful compilation in baseline
    except SyntaxError:
        assert False  # This shouldn't happen in the baseline
    except TypeError as e:
        assert str(e) == "unsupported operand type(s) for |: 'float' and 'RegexFlag'"  # Check the exact mutant behavior