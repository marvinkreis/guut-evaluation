from string_utils.validation import is_json

def test__is_json():
    """
    Test the functionality of the is_json function. The input is a properly formatted JSON string.
    If the code is correct, it will return True. However, with the mutant, the logic has been inverted,
    leading it to return False for valid JSON strings.
    """
    output = is_json('{"name": "Peter", "age": 30}')
    assert output is True