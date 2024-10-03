from string_utils._regex import HTML_RE

def test__html_regex_doctype_handling():
    """
    Test if the mutant correctly recognizes a DOCTYPE declaration in HTML. 
    The input '<!DOCTYPE html>' should match in the baseline but fail in the mutant due to the incorrect flag combination.
    """
    test_string = "<!DOCTYPE html>"
    is_matching = HTML_RE.match(test_string)

    assert is_matching is not None, "Baseline should match DOCTYPE, but it doesn't."