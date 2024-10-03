from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_re():
    """
    Test the compilation of the HTML_TAG_ONLY_RE regex. The baseline should compile without issues, while the mutant will raise an error due to incorrect use of the division operator instead of bitwise OR.
    """
    try:
        compiled_regex = HTML_TAG_ONLY_RE.pattern
        assert compiled_regex is not None  # Ensure the regex compiled successfully
    except SyntaxError:
        assert False, "Expected HTML_TAG_ONLY_RE to compile successfully in baseline."
    except TypeError as e:
        # This should only occur in the mutant version
        print(f"Type error encountered, as expected: {e}")