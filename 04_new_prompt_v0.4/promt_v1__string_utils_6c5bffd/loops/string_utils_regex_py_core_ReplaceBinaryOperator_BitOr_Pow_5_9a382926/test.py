from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_re_doctype():
    """
    Test to verify that the regex correctly matches the <!DOCTYPE html> declaration.
    The baseline should recognize it as a valid declaration, while the mutant should fail 
    due to the incorrect implementation. This distinction will confirm the mutant's behavior difference.
    """
    
    doctype_input = "<!DOCTYPE html>"
    match = HTML_TAG_ONLY_RE.match(doctype_input)
    assert match is not None, "Expected a match for the doctype in baseline."