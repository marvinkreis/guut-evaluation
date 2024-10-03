from string_utils.validation import is_json

def test__is_json():
    """
    Test whether the function correctly identifies a valid JSON string.
    The input is '{"name": "Peter"}', which should return true for a valid JSON.
    The baseline returns True, while the mutant incorrectly returns False because of the logical change.
    """
    output = is_json('{"name": "Peter"}')
    assert output == True, f"Expected True, got {output}"