from string_utils.manipulation import strip_html

def test__strip_html_mutant_killer():
    """
    Test the strip_html function to distinguish between baseline and mutant behavior.
    We expect that stripping the HTML content without keeping tag content returns 
    missing the tag content, while the mutant retains it due to its change in the default parameter.
    """
    output = strip_html('<p>Hello</p> World')
    assert output == ' World'  # This should pass in the baseline