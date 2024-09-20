from string_utils._regex import HTML_RE

def test__html_regex_correct():
    """Test the valid HTML regex from the correct implementation."""
    test_html = "<html><body><h1>Hello World</h1></body></html>"
    correct_output = HTML_RE.match(test_html)
    assert correct_output is not None, "The correct HTML_RE should match valid HTML."

def test__html_regex_mutant():
    """Simulate a failed import from the mutant due to incorrect regex flags."""
    try:
        from mutant.string_utils._regex import HTML_RE
        raise AssertionError("Mutant should not compile but it did!")
    except Exception as e:
        assert isinstance(e, ValueError), "The mutant should raise a ValueError for incompatible flags."