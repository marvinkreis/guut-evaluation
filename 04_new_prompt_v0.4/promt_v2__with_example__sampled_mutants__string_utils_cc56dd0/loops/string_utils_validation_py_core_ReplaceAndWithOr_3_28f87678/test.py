from string_utils.validation import is_url

def test_is_url_mutant_killing():
    """
    Test the is_url function with an invalid URL that begins with a valid scheme.
    The mutant will return True due to the change from 'and' to 'or', while the baseline will return False.
    """
    output = is_url("http://invalid-url", allowed_schemes=["http"])
    assert output == False, f"Expected False, got {output}"