from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_multiline():
    """
    Test matching an HTML tag that spans multiple lines. The baseline should match this tag correctly,
    while the mutant should fail due to the incompatibility of ASCII and UNICODE flags in the regex.
    The input is a multiline HTML string with a <div> tag.
    """
    html_string = """
    <div>
        <p>Hello, world!</p>
    </div>
    """
    # This should execute without raising an error on the baseline
    output = HTML_TAG_ONLY_RE.search(html_string)
    assert output is not None  # This assertion should pass for the baseline