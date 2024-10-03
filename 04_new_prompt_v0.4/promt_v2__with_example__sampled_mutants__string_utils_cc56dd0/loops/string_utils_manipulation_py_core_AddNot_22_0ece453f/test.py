from string_utils.manipulation import strip_html

def test_strip_html_mutant_killing():
    """
    Test the strip_html function with an empty string.
    The baseline should return an empty string, while the mutant will raise an InvalidInputError
    due to the faulty input validation.
    """
    output = strip_html('')  # This should pass in the baseline
    assert output == '', f"Expected '', but got '{output}'"