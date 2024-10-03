from string_utils.validation import is_json

def test_is_json_mutant_killing():
    """
    Test the is_json function with a valid JSON string. The baseline will return True,
    indicating the string is valid JSON, while the mutant will incorrectly return False.
    """
    output = is_json('{"name": "Peter"}')
    assert output == True, f"Expected True but got {output}"