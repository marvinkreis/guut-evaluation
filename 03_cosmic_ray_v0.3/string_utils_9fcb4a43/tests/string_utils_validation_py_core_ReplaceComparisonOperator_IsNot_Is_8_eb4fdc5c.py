from string_utils.validation import is_json

def test__is_json():
    """The mutant changes the logic, so valid JSON strings should return False."""
    test_input = '{"name": "Peter"}'
    output = is_json(test_input)
    assert output == True, "is_json must return True for valid JSON strings"