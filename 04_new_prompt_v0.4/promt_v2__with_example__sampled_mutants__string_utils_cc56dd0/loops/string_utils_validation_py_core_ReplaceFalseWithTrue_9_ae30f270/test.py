from string_utils.validation import is_json

def test_is_json_mutant_killing():
    """
    Test the is_json function using an invalid JSON string. The mutant will incorrectly return true,
    while the baseline will correctly return false.
    """
    output = is_json("{nope}")
    assert output == False, f"Expected False, got {output}"