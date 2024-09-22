from string_utils._regex import HTML_TAG_ONLY_RE

def test_HTML_TAG_ONLY_RE():
    # Test to ensure HTML_TAG_ONLY_RE compiles correctly
    try:
        # This will cause an assertion error if the regex doesn't compile correctly.
        assert HTML_TAG_ONLY_RE is not None
        # Additionally, we can check if it matches a specific html tag
        test_string = "<div></div>"
        assert HTML_TAG_ONLY_RE.match(test_string) is not None
    except Exception as e:
        # If an exception occurs, it means the regex failed to compile or didn't match properly
        raise AssertionError("The regex failed to compile or match: {}".format(e))