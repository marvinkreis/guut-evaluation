from string_utils._regex import HTML_RE

def test__html_re():
    # The original HTML_RE should match a simple HTML tag
    assert HTML_RE.match("<div>") is not None
    # A complex HTML with content should also match
    assert HTML_RE.match("<div>Some content</div>") is not None
    # An HTML comment should match
    assert HTML_RE.match("<!-- This is a comment -->") is not None
    # The mutant would fail on this input due to changes in regex combining
    assert HTML_RE.match("<p>Paragraph</p>") is not None
    # This input might be obstructed by the changes in regex behavior caused by the mutant
    assert HTML_RE.match("<!doctype html>") is not None