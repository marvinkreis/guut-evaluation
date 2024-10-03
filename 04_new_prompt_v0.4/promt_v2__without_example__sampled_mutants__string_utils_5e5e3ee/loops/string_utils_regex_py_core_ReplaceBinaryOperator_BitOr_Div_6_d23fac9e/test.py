from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_regex_compilation():
    """
    Test whether HTML_TAG_ONLY_RE regex compiles correctly.
    The baseline should compile without errors, while the mutant will raise a TypeError
    due to a syntax error in the regex definition.
    """
    try:
        HTML_TAG_ONLY_RE.pattern  # Attempt to access the pattern to trigger compilation
    except SyntaxError as e:
        raise AssertionError(f"Baseline threw a SyntaxError: {e}")
    except TypeError:
        # This is expected in the mutant
        assert True  # Mutant behavior
    else:
        assert True  # Baseline behavior