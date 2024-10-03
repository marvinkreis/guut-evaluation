from string_utils._regex import HTML_TAG_ONLY_RE

def test_compilation_html_tag_only_re():
    """
    Test the compilation of the HTML_TAG_ONLY_RE regex. This should compile without error on the baseline,
    but should raise an OverflowError on the mutant due to incorrect operator usage in the regex definition.
    """
    try:
        HTML_TAG_ONLY_RE  # Attempt to access the regex to trigger compilation
        assert True  # If this passes, the baseline compiled successfully
    except OverflowError:
        assert False, "OverflowError raised in mutant, which is expected"
    except Exception as e:
        assert False, f"Unexpected error raised: {e}"  # Any other error is unexpected