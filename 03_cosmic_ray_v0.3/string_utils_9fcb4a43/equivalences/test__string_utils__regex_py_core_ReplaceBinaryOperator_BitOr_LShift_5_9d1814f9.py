from string_utils._regex import HTML_TAG_ONLY_RE

def test():
    assert HTML_TAG_ONLY_RE.match("<P>")
