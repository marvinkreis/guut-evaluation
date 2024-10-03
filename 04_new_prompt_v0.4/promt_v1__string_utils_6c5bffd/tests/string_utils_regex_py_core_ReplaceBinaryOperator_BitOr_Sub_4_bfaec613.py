from string_utils._regex import HTML_RE

def test__HTML_RE():
    """
    Test the HTML_RE regex for matching valid HTML input. The baseline should successfully match the input containing multiple lines of HTML due to the presence of the re.DOTALL flag. The mutant, however, raises a ValueError because it incorrectly attempts to modify the flags. This test verifies that the mutant fails where the baseline succeeds.
    """
    html_input = """<div>
                        <p>Hello World</p>
                    </div>"""
    output = HTML_RE.match(html_input)
    assert output is not None