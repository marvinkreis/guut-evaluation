from string_utils.validation import is_json

def test__is_json_invalid_input():
    """
    Test to check that is_json returns False for invalid JSON input.
    The baseline should handle a malformed JSON string correctly, returning False.
    The mutant is expected to raise a NameError because it references an undefined
    CosmicRayTestingException, which will lead to a different outcome.
    """
    invalid_json = '{"name": "Peter",}'  # Malformed JSON (trailing comma)

    # This should pass on baseline
    output = is_json(invalid_json)
    assert output is False