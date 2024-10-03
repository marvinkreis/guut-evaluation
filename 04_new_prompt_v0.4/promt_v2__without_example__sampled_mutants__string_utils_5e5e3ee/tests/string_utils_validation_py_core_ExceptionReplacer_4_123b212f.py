from string_utils.validation import is_json

def test__is_json_invalid():
    """
    Test whether the is_json function correctly identifies an invalid JSON string.
    The input '{nope}' is invalid JSON and should return False in the baseline.
    The mutant does not handle exception correctly, leading to a NameError instead.
    """
    output = is_json('{nope}')
    assert output == False