from string_utils.validation import is_snake_case

def test__is_snake_case():
    """
    Test that an empty string is not accepted as a valid snake_case.
    The input should return False in the baseline implementation,
    and True in the mutant implementation, hence confirming the mutant's error.
    """
    output = is_snake_case('')
    assert output == False  # Expected output for baseline